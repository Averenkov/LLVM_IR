# Задание для AI-агента (Codex): реализация эвристик агрегации pass sequences

## 1. Контекст проекта

Дипломный проект по оптимизации LLVM IR. Pipeline:

1. **Этап 1** (готов): построение датасета LLVM bitcode-функций.
2. **Этап 2** (готов): поиск хорошей последовательности passes для каждой функции (CEM, Random Search).
3. **Этап 3** (текущая задача): агрегация per-function pass sequences в одну общую sequence для translation unit.

Уже реализованы 4 эвристики этапа 3: `greedy_consensus`, `dag_longest_path`, `beam_search`, `weighted_toposort`. Нужно добавить 10 новых эвристик + одну главную (HPP-Eades-TopK).

## 2. Цель задания

Реализовать в существующей кодовой базе:
- 10 новых эвристик агрегации.
- 1 главную композитную эвристику `HPP-Eades-TopK`.
- Единый CLI/API для их запуска.
- Систему метрик и сравнения.
- Тесты.

## 3. Структура входных данных

Считай, что доступны следующие сущности (если их API отличается — адаптируй, но не ломай существующий):

```python
@dataclass
class PerFunctionResult:
    function_id: str
    sequence: list[str]          # список passes
    baseline_size: int
    best_size: int
    @property
    def delta(self) -> int:      # baseline_size - best_size; может быть < 0
        ...

@dataclass
class Dataset:
    results: list[PerFunctionResult]
    translation_unit_path: str   # путь к .bc для TU-валидации
    baseline_text_size: int
    oz_text_size: int
```

Граф уже умеешь строить (есть в проекте). Если функции построения нет — добавь:

```python
def build_pass_graph(results: list[PerFunctionResult]) -> PassGraph:
    """
    Узлы: passes.
    Рёбра: p_i -> p_j, если p_i раньше p_j хотя бы в одной sequence.
    Веса: count_weight, delta_weight.
    """
```

## 4. Что реализовать

### 4.1. Общая инфраструктура

Создай модуль `heuristics/aggregation/`:

```
heuristics/aggregation/
    __init__.py
    base.py                  # AggregationHeuristic (ABC), AggregationResult
    stats.py                 # node-level статистики
    graph_utils.py           # SCC, Eades, нормализация весов
    tu_eval.py               # запуск opt+llc+llvm-size с кэшем
    wfas_eades.py
    scc_ordering.py
    pagerank_ordering.py
    position_median.py
    hpp.py
    beam_diversity.py
    cluster_aware.py
    ilp_arrangement.py
    markov_hitting.py
    voting_ensemble.py
    hpp_eades_topk.py        # ГЛАВНАЯ
    registry.py              # реестр всех эвристик
```

### 4.2. Базовый интерфейс

```python
class AggregationHeuristic(ABC):
    name: str
    supports_topk: bool = False

    @abstractmethod
    def aggregate(
        self,
        dataset: Dataset,
        graph: PassGraph,
        config: dict,
    ) -> AggregationResult: ...

@dataclass
class AggregationResult:
    sequences: list[list[str]]      # top-K; для не-topk len==1
    chosen_sequence: list[str]
    chosen_prefix_length: int
    extra: dict                     # метрики, диагностика
```

### 4.3. Эвристики для реализации

Для каждой реализуй отдельный класс-наследник `AggregationHeuristic`. Описания, формулы и pseudocode — ниже. Параметры по умолчанию делай разумными, все вынеси в `config` с возможностью override через CLI.

---

#### Эвристика 1. WFAS-Eades

Файл: `wfas_eades.py`, класс `WFASEades`.

Алгоритм:
1. Вес ребра: `w[i,j] = alpha * norm(w_delta[i,j]) + (1-alpha) * norm(w_count[i,j])`.
2. Eades-Lin-Smyth:
   - Пока граф не пуст:
     - Удалить все sinks → в стек `S2`.
     - Удалить все sources → в очередь `S1`.
     - Иначе выбрать `v* = argmax(out_weight - in_weight)`, добавить в `S1`.
3. `T = S1 + reverse(S2)`.
4. Best-prefix: посчитать prefix-score, вернуть префикс с максимальным prefix-score (для extra).

Параметры: `alpha=0.5`.
Сложность: O(V+E).

---

#### Эвристика 2. SCC-Ordering

Файл: `scc_ordering.py`, класс `SCCOrdering`.

Алгоритм:
1. Tarjan → SCC.
2. Condensation DAG → топосортировка с приоритетом по `sum(gain(p) for p in SCC)`.
3. Внутри SCC размера >1: сортировка по `gain(p)/freq(p)`, tie-break — `net(p)`.
4. Конкатенация. Cut-points = границы SCC.

Сложность: O(V+E).

---

#### Эвристика 3. PageRank Ordering

Файл: `pagerank_ordering.py`, класс `PageRankOrdering`.

Алгоритм:
1. Transition: `M[j,i] = w_delta[j,i] / sum_k w_delta[j,k]`.
2. Personalization: `pi(p) ∝ gain(p)`.
3. Итерации (damping `d=0.85`, до 50 итераций или eps=1e-6).
4. `r_in` на прямом графе, `r_out` на reverse-графе.
5. Сортировка по `r_in - r_out` по убыванию.

Параметры: `damping=0.85, max_iter=50, eps=1e-6`.

---

#### Эвристика 4. Position-Median Ordering

Файл: `position_median.py`, класс `PositionMedianOrdering`.

Алгоритм:
1. Для каждого `p` — список нормализованных позиций `pos_f(p)/k_f` с весами `Δ_f`.
2. Взвешенная медиана.
3. Опциональный фильтр: если `std(normalized_pos) > tau` — выбрасывать pass.
4. Сортировка по медиане, tie-break — gain.

Параметры: `tau=0.35, use_filter=True`.

---

#### Эвристика 5. HPP (Harmful-Pass Penalized)

Файл: `hpp.py`, класс `HPPHeuristic`.

Алгоритм:
1. `harm(p) = sum |Δ_f| (Δ_f<0, p∈S_f) - λ * sum Δ_f (Δ_f>0, p∈S_f)`.
2. `s(p) = w_out_delta(p) - w_in_delta(p) - β * normalize(harm(p))`.
3. Pruning: убрать `p` если `s(p) < θ`.
4. Запустить базовый ordering (по умолчанию — WFAS-Eades) на подграфе.

Параметры: `lambda_=0.1, beta=1.0, theta=-inf, base_ordering="eades"`.

---

#### Эвристика 6. Beam Search with Diversity

Файл: `beam_diversity.py`, класс `BeamSearchDiversity`. `supports_topk=True`.

Алгоритм:
1. Beam ширины B; состояние `(prefix, score, used_set)`.
2. На шаге для каждого beam: для каждого `p ∉ used` посчитать
   `Δs = sum w_delta[q→p] - sum w_delta[p→q]` по `q∈prefix`.
3. Diversity penalty: `γ * mean Jaccard(used∪{p}, b'.used)` по другим beam.
4. Top-B по score; в конце top-K.

Параметры: `beam_width=16, top_k=4, gamma=0.5`.

---

#### Эвристика 7. Cluster-aware Aggregation

Файл: `cluster_aware.py`, класс `ClusterAwareAggregation`. `supports_topk=True`.

Алгоритм:
1. Каждая `S_f` → multi-hot vector над passes.
2. K-means (`K=4` по умолчанию, конфигурируемо). Использовать sklearn если доступен, иначе собственный k-means на cosine.
3. На каждый кластер строить свой граф, запускать WFAS-Eades → `T_k`.
4. `rank(p) = sum_k W_k * rank_T_k(p)`, `W_k = sum Δ_f`.
5. Сортировка по rank. Также возвращай top-K = по одной sequence на кластер.

Параметры: `n_clusters=4, base_ordering="eades"`.

---

#### Эвристика 8. ILP-relaxed Linear Arrangement

Файл: `ilp_arrangement.py`, класс `ILPLinearArrangement`.

Алгоритм:
1. Использовать PuLP или scipy.optimize.linprog (для LP) либо `python-mip`. Если ни одного solver-а нет — обернуть в `ImportError` и пометить эвристику как `optional`.
2. LP-релаксация FAS:
   - `x_ij ∈ [0,1]`, `x_ij + x_ji = 1`.
   - Транзитивность `x_ij + x_jk + x_ki ≤ 2` (если число passes ≤ 50, иначе пропускать часть).
   - Максимизировать `sum w_delta[i,j] * x_ij`.
3. Сортировать passes по `sum_j x_ij`.

Параметры: `max_nodes_for_transitivity=50, solver="cbc"`.

---

#### Эвристика 9. Markov Hitting Time

Файл: `markov_hitting.py`, класс `MarkovHittingOrdering`.

Алгоритм:
1. Transition `P[i,j] = w_delta[i,j] / sum_k w_delta[i,k]`. Для строк-нулей — uniform.
2. Mean hitting time из uniform start: численно через итерации или через линейную систему `(I - P_{-j}) h = 1` для каждой `j` (по умолчанию итеративно).
3. Сортировка по возрастанию `h(p)`.

Параметры: `max_iter=200, eps=1e-6`.

---

#### Эвристика 10. Voting Ensemble

Файл: `voting_ensemble.py`, класс `VotingEnsemble`.

Алгоритм:
1. Принимает список других эвристик (имена из registry).
2. Borda count: `score(p) = sum_r (n - rank_r(p))`.
3. Сортировка по score.

Параметры: `voters=["wfas_eades","scc_ordering","pagerank","position_median","hpp"]`.

---

#### ГЛАВНАЯ. HPP-Eades-TopK

Файл: `hpp_eades_topk.py`, класс `HPPEadesTopK`. `supports_topk=True`.

Алгоритм — см. псевдокод ниже. Внутри композирует:
- node stats,
- нормализацию весов с `alpha`,
- HPP pruning,
- WFAS-Eades на подграфе,
- diversified beam search seeded by Eades order,
- TU-валидацию на prefix grid с memoization.

Параметры по умолчанию:
```python
{
  "alpha": 0.5,
  "beta": 1.0,
  "lambda_": 0.1,
  "theta": float("-inf"),
  "beam_width": 16,
  "top_k": 4,
  "gamma": 0.5,
  "prefix_grid": [0.3, 0.5, 0.7, 1.0],
}
```

Псевдокод (см. ниже раздел 6) реализовать строго.

---

### 4.4. TU evaluation модуль

`tu_eval.py`:

```python
class TUEvaluator:
    def __init__(self, tu_path, opt_bin, llc_bin, size_bin, cache_dir):
        ...
    def evaluate(self, passes: list[str], timeout: float = 60.0) -> EvalResult:
        """
        opt -passes=<passes> tu.bc -o tmp.bc
        llc tmp.bc -o tmp.s
        llvm-size tmp.s -> .text size
        """
        ...

@dataclass
class EvalResult:
    text_size: int | None       # None если failure
    success: bool
    error: str | None
    elapsed_s: float
```

Требования:
- Кэш по SHA-1(`passes_str + tu_hash`) на диск (`.json` или sqlite).
- Безопасный запуск (timeout, capture stderr).
- Учёт failures для метрики `FailRate`.

### 4.5. Метрики и сравнение

`heuristics/aggregation/metrics.py`:

```python
@dataclass
class HeuristicMetrics:
    name: str
    graph_score_delta: float
    fas_weight: float
    coverage: float
    final_size: int | None
    best_prefix_size: int | None
    delta_vs_oz: int | None
    norm_best: float | None
    fail_rate: float
    beat_oz: bool
    tu_evals: int
    wallclock_s: float
```

`heuristics/aggregation/compare.py`:

```python
def run_comparison(
    datasets: list[Dataset],
    heuristics: list[AggregationHeuristic],
    evaluator_factory,
    output_path: str,
):
    """
    Запускает все эвристики на всех TU.
    Сохраняет CSV + JSON-отчёт.
    Считает gm(NormBest), BeatOzRate, Wilcoxon vs greedy_consensus, vs -Oz.
    """
```

Должна быть таблица вида: rows=heuristics, columns=metrics + p-values Wilcoxon.

### 4.6. CLI

`scripts/run_aggregation.py`:

```
python -m scripts.run_aggregation \
    --dataset <path> \
    --heuristic hpp_eades_topk \
    --config configs/hpp_eades_topk.yaml \
    --output runs/<exp_id>/

python -m scripts.run_aggregation --all --compare ...
```

Поддерживать:
- `--heuristic <name>` или `--all`.
- `--config <yaml>` override параметров.
- `--no-tu-eval` (только graph-score).
- `--top-k-validate` (для эвристик с `supports_topk`).
- `--cache-dir`.

### 4.7. Тесты

`tests/aggregation/`:
- Unit-тесты на каждую эвристику с маленьким синтетическим графом (3–7 passes).
- Property-тесты:
  - выходная sequence — перестановка подмножества passes;
  - WFAS-Eades возвращает топологический порядок, если граф — DAG;
  - Voting сохраняет порядок при идентичных voters.
- Тест на кэш `TUEvaluator` (мок `opt`/`llc`/`llvm-size`).
- Тест на корректность TopK (no duplicates, размер ≤ K).

## 5. Требования к коду

- Python 3.10+, type hints везде.
- Стиль: `ruff` + `black`.
- Без скрытых глобальных состояний.
- Все случайные выборы — через явный `random.Random(seed)` или `np.random.default_rng(seed)`.
- Логирование через `logging`, по умолчанию INFO.
- Никаких ML-моделей; только классические алгоритмы.
- Все эвристики должны корректно отрабатывать:
  - пустой граф (вернуть пустую sequence),
  - граф из одного узла,
  - граф с self-loops (игнорировать self-loops),
  - дисконнектные компоненты.
- Для ILP — graceful fallback при отсутствии solver-а.

## 6. Псевдокод главной эвристики (реализовать буквально)

```text
Algorithm: HPP-Eades-TopK
Input:
  S, Delta, alpha, beta, theta, lambda_, B, K, prefix_grid, gamma

Step 1. Node stats
  for each p:
    freq[p], gain[p], harm0[p], harm[p] = harm0[p] - lambda_*gain[p]

Step 2. Edge weights
  ŵ_c, ŵ_δ = normalize(w_c, w_δ)
  w[i,j] = alpha*ŵ_δ[i,j] + (1-alpha)*ŵ_c[i,j]

Step 3. Prune harmful
  s_raw[p] = Σ_j w[p,j] - Σ_j w[j,p] - beta*normalize(harm[p])
  V_keep   = { p : s_raw[p] >= theta }
  G' = induced subgraph

Step 4. WFAS-Eades on G'
  Eades algorithm → T_eades

Step 5. Diversified beam search seeded by T_eades
  beams = [ (prefix=[], score=0, used=∅) ] × B
  for step in 1..|V_keep|:
    expand each beam by all unused p;
    score expansion by Δs - γ*Jaccard-diversity;
    keep top-B
  top_K = top_K by score from beams

Step 6. TU validation on prefix grid
  best = (None, +∞)
  for T in top_K + [T_eades]:
    for frac in prefix_grid:
      L = ceil(frac*len(T))
      prefix = T[:L]
      size = TUEvaluator.evaluate(prefix)
      if size < best.size: best = (T, L, size)
  return best
```

## 7. Порядок реализации (важно)

Реализуй в этом порядке, чтобы было удобно мерджить и ревьюить по частям:

1. `base.py`, `stats.py`, `graph_utils.py`, `tu_eval.py`, `registry.py`, `metrics.py`, `compare.py`.
2. Эвристика 1 (WFAS-Eades) + тесты — это база для главной и для 5, 7.
3. Эвристика 5 (HPP) + тесты.
4. Эвристика 6 (Beam+Diversity) + тесты.
5. **Главная: HPP-Eades-TopK** + тесты + интеграционный тест с моком `TUEvaluator`.
6. Эвристики 2, 3, 4, 9 (быстрые, независимые).
7. Эвристика 7 (Cluster-aware).
8. Эвристика 10 (Voting).
9. Эвристика 8 (ILP, последняя из-за зависимости).
10. CLI + сравнение.

## 8. Definition of Done

- Все 10 + 1 эвристик имплементированы и зарегистрированы в `registry.py`.
- `python -m scripts.run_aggregation --all --compare --dataset <demo>` отрабатывает end-to-end на демо-датасете без ошибок.
- Все unit-тесты проходят (`pytest tests/aggregation/`).
- `ruff check` и `black --check` без ошибок.
- README в `heuristics/aggregation/README.md` с:
  - кратким описанием каждой эвристики,
  - таблицей параметров по умолчанию,
  - примерами CLI-вызовов,
  - описанием формата output-отчёта.
- Сохраняется CSV/JSON с метриками вида:
  `heuristic, graph_score_delta, fas_weight, coverage, final_size, best_prefix_size, delta_vs_oz, norm_best, fail_rate, beat_oz, tu_evals, wallclock_s`.

## 9. Что НЕ нужно делать

- Не трогай этапы 1 и 2 проекта.
- Не меняй существующие 4 эвристики (`greedy_consensus`, `dag_longest_path`, `beam_search`, `weighted_toposort`) — только обвяжи их общим интерфейсом, если он отличается.
- Не вводи ML-модели.
- Не делай тяжёлых зависимостей обязательными: `sklearn`, `pulp`, `scipy` — optional, при их отсутствии соответствующие эвристики помечаются `unavailable`, но импорт пакета не падает.

## 10. Что вернуть в конце

- Diff/PR со всеми изменениями.
- Список добавленных файлов.
- Команды для воспроизведения demo-сравнения.
- Краткий отчёт: какие эвристики работают, какие требуют доустановки зависимостей, известные ограничения.
