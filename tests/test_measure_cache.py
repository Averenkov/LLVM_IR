"""Tests for the persistent measurement cache."""

import tempfile
import unittest
from pathlib import Path

from llvm_ir.stages.translation_unit.measure_cache import MeasureCache, tu_key


class MeasureCacheTests(unittest.TestCase):
    def _bitcode(self, tmp: Path, content: bytes = b"hello-bitcode") -> Path:
        p = tmp / "tu.bc"
        p.write_bytes(content)
        return p

    def test_put_get_and_persist_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            bc = self._bitcode(tmp)
            cache_dir = tmp / "cache"
            c1 = MeasureCache(cache_dir, bc)
            self.assertIsNone(c1.get_size(("a", "b")))
            c1.put(("a", "b"), 1000, 42)
            c1.flush()
            # New instance on the same TU reloads persisted values.
            c2 = MeasureCache(cache_dir, bc)
            self.assertEqual(c2.get_size(("a", "b")), 1000)
            self.assertEqual(c2.get_instr(("a", "b")), 42)

    def test_size_without_instr(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            c = MeasureCache(tmp / "cache", self._bitcode(tmp))
            c.put(("x",), 500)
            self.assertEqual(c.get_size(("x",)), 500)
            self.assertIsNone(c.get_instr(("x",)))

    def test_distinct_tu_distinct_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            bc1 = self._bitcode(tmp / "a" if False else tmp, b"AAAA")
            # two different contents -> different keys -> isolated caches
            (tmp / "tu2.bc").write_bytes(b"BBBB")
            bc2 = tmp / "tu2.bc"
            self.assertNotEqual(tu_key(bc1), tu_key(bc2))
            cache_dir = tmp / "cache"
            c1 = MeasureCache(cache_dir, bc1); c1.put(("p",), 1); c1.flush()
            c2 = MeasureCache(cache_dir, bc2)
            self.assertIsNone(c2.get_size(("p",)))  # isolated by TU content

    def test_disabled_cache_noop(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            c = MeasureCache(None, self._bitcode(Path(d)))
            c.put(("a",), 10)
            c.flush()
            self.assertIsNone(c.get_size(("a",)))
            self.assertFalse(c.enabled)


if __name__ == "__main__":
    unittest.main()
