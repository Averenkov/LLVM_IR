"""Tests for the bucket-layered DAG heuristic."""

import unittest

from llvm_ir.heuristics.translation_unit.bucket_dag import (
    BucketDagConfig,
    bucket_layer_top_paths,
    build_teacher_graph,
    node_out_weights,
)
from llvm_ir.stages.translation_unit.evaluate_bucket_dag_teacher import (
    edge_marginals_from_cache,
)
from llvm_ir.stages.translation_unit.graph.order_graph import PassOrderGraph


def _graph(edges: dict[tuple[str, str], int]) -> PassOrderGraph:
    graph = PassOrderGraph(benchmark="demo")
    for (source, target), weight in edges.items():
        graph.nodes.add(source)
        graph.nodes.add(target)
        graph.edge_counts[(source, target)] = weight
    return graph


class BucketDagTests(unittest.TestCase):
    def test_out_weights_sum_outgoing_only(self) -> None:
        graph = _graph({("a", "b"): 5, ("a", "c"): 3, ("b", "c"): 2})
        weights = node_out_weights(graph)
        self.assertEqual(weights["a"], 8)
        self.assertEqual(weights["b"], 2)
        self.assertEqual(weights["c"], 0)

    def test_edges_only_forward_between_buckets(self) -> None:
        # Out-weights: a=10, b=6, c=1, d=0 -> ranking a,b,c,d.
        # chunk_size=2 -> bucket0={a,b}, bucket1={c,d}.
        # Backward edge c->a (bucket1->bucket0) and within-bucket a->b must be dropped;
        # forward edge a->c (bucket0->bucket1) must be kept.
        graph = _graph(
            {
                ("a", "b"): 4,
                ("a", "c"): 6,
                ("b", "c"): 6,
                ("c", "a"): 1,
                ("c", "d"): 0,
            }
        )
        paths = bucket_layer_top_paths(
            graph, config=BucketDagConfig(chunk_size=2, max_length=12, min_edge_weight=1), top_k=10
        )
        self.assertTrue(paths)
        for path in paths:
            # No path may contain a backward (c before a) or within-bucket (a before b) hop.
            for first, second in zip(path, path[1:]):
                self.assertNotEqual((first, second), ("c", "a"))
                self.assertNotEqual((first, second), ("a", "b"))
        # The heaviest path should be a -> c (forward, weight 6).
        self.assertIn(["a", "c"], paths)

    def test_construction_is_acyclic_even_with_cycles(self) -> None:
        # A 3-cycle a->b->c->a; bucketing must still yield only forward edges.
        graph = _graph({("a", "b"): 3, ("b", "c"): 2, ("c", "a"): 5})
        paths = bucket_layer_top_paths(
            graph, config=BucketDagConfig(chunk_size=1, max_length=12, min_edge_weight=1), top_k=20
        )
        self.assertTrue(paths)
        # No path repeats a node (acyclic) and respects bucket ordering.
        for path in paths:
            self.assertEqual(len(path), len(set(path)))

    def test_respects_top_k_and_max_length(self) -> None:
        graph = _graph({("a", "b"): 5, ("a", "c"): 4, ("b", "c"): 3, ("a", "d"): 2})
        paths = bucket_layer_top_paths(
            graph, config=BucketDagConfig(chunk_size=1, max_length=2, min_edge_weight=1), top_k=3
        )
        self.assertLessEqual(len(paths), 3)
        for path in paths:
            self.assertLessEqual(len(path), 2)

    def test_empty_graph(self) -> None:
        self.assertEqual(bucket_layer_top_paths(PassOrderGraph(benchmark="x"), top_k=10), [])


class BucketDagTeacherTests(unittest.TestCase):
    def test_edge_marginals_from_cache(self) -> None:
        # Path [a, b, c] with cached sizes 1000 -> 900 (a) -> 850 (b) -> 870 (c).
        prefix_cache = {
            (): {"size": 1000, "error": ""},
            ("a",): {"size": 900, "error": ""},
            ("a", "b"): {"size": 850, "error": ""},
            ("a", "b", "c"): {"size": 870, "error": ""},
        }
        means = edge_marginals_from_cache([["a", "b", "c"]], prefix_cache)
        # a->b marginal = 900-850 = 50; b->c = 850-870 = -20. First pass 'a' has no edge.
        self.assertEqual(means[("a", "b")], 50)
        self.assertEqual(means[("b", "c")], -20)
        self.assertNotIn(("", "a"), means)

    def test_marginals_stop_at_crashed_prefix(self) -> None:
        prefix_cache = {
            (): {"size": 1000, "error": ""},
            ("a",): {"size": 900, "error": ""},
            ("a", "b"): {"size": 900, "error": "boom"},
        }
        means = edge_marginals_from_cache([["a", "b", "c"]], prefix_cache)
        self.assertEqual(means, {})  # crash at b stops attribution

    def test_teacher_graph_keeps_measured_clamps_and_rescales(self) -> None:
        graph = PassOrderGraph(benchmark="d")
        for (s, t), w in {("a", "b"): 10, ("a", "c"): 4, ("b", "c"): 6}.items():
            graph.nodes.update((s, t))
            graph.edge_counts[(s, t)] = w
        teacher = build_teacher_graph(graph, {("a", "b"): 200.0, ("a", "c"): -5.0})
        self.assertEqual(teacher.edge_counts[("a", "b")], 200)  # measured kept
        self.assertEqual(teacher.edge_counts[("a", "c")], 0)  # negative clamped
        # scale = mean([200]) / mean(fn weights of positive-measured = [10]) = 20
        self.assertEqual(teacher.edge_counts[("b", "c")], 120)  # 6 * 20, rescaled


if __name__ == "__main__":
    unittest.main()
