# LLVM_IR

Инструменты для экспериментов с последовательностями LLVM optimization pass-ов:
построение per-function bitcode-датасета, подбор pass sequence для отдельных
функций и агрегация найденных последовательностей в общий порядок pass-ов для
единицы трансляции.

Проект строится как пайплайн из трёх частей:

1. `src/llvm_ir/stages/dataset/` - построение per-function датасета из
   CompilerGym и опциональный отбор top 20% функций по размеру LLVM IR.
2. `src/llvm_ir/stages/function_search/` - подбор оптимальной
   последовательности pass-ов для каждой функции. Поддерживаются CEM, Random
   Search и опциональное сравнение с PPO из `llvm-minimizer`.
3. `src/llvm_ir/stages/translation_unit/` - построение pass-order graph,
   запуск эвристик для выбора общего TU-level порядка pass-ов и реальная
   проверка найденных последовательностей на whole-translation-unit bitcode.

Отдельный исследовательский слой `src/llvm_ir/heuristics/aggregation/`
сравнивает более сильные эвристики агрегации per-function последовательностей:
weighted feedback arc, SCC ordering, PageRank, position median, harmful-pass
pruning, diversity beam search, cluster-aware aggregation, ILP fallback,
Markov-style ordering, voting ensemble и комбинированный `hpp_eades_topk`.

Первый шаг stage 3 - построение pass-order graph по найденным per-function
последовательностям. Для каждого benchmark-а строится ориентированный граф:
ребро `p_i -> p_j` означает, что в хотя бы одной лучшей последовательности
функции benchmark-а pass `p_i` встречался раньше pass `p_j`.

В корне пакета оставлены совместимые модули для старых импортов, но основная
логика разложена по stage-каталогам: dataset, function search и translation
unit evaluation.

## Установка

Самый простой путь на Ubuntu 24.04 и близких Debian/Ubuntu-системах:

```bash
scripts/setup_ubuntu.sh
source .venv/bin/activate
```

Скрипт устанавливает системные зависимости через `apt`, создаёт локальный
`.venv`, ставит пакет в editable-режиме, проверяет unit-тесты, LLVM CLI tools и
CompilerGym `llvm-v0`.

Системные зависимости, которые нужны проекту:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  clang \
  cmake \
  llvm \
  python3-dev \
  python3-pip \
  python3.12-venv \
  wget
```

Для CompilerGym на новых Ubuntu нужен compatibility-пакет `libtinfo5`, потому
что bundled `compiler_gym-llvm-service` собран против `libtinfo.so.5`:

```bash
wget -O /tmp/libtinfo5.deb \
  http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
sudo dpkg -i /tmp/libtinfo5.deb
```

Python-окружение без CompilerGym, достаточное для тестов, pass search по уже
готовым `.bc` и aggregation/evaluation стадий:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

CompilerGym `0.2.5` старый и на Python 3.12 не ставится обычным
`pip install compiler_gym` из-за устаревших pins (`grpcio`, `gym`, `protobuf`,
`networkx`). Для dataset stage используйте команды из `scripts/setup_ubuntu.sh`
или эквивалентную ручную установку:

```bash
python -m pip install compiler_gym==0.2.5 --no-deps
python -m pip install \
  absl-py deprecated docker fasteners grpcio gym==0.26.2 humanize \
  loop-tool-py 'networkx<3' 'numpy<2' protobuf==3.20.3 \
  pydantic requests tabulate
```

Проверка окружения:

```bash
python -m unittest discover -s tests -v
python - <<'PYCHECK'
import shutil
missing = [tool for tool in ("opt", "llc", "llvm-size", "llvm-dis", "llvm-extract", "llvm-as") if shutil.which(tool) is None]
if missing:
    raise SystemExit("Missing LLVM tools: " + ", ".join(missing))
print("LLVM tools: ok")
PYCHECK
python - <<'PYCHECK'
import compiler_gym
env = compiler_gym.make("llvm-v0", disable_env_checker=True)
print(f"CompilerGym llvm-v0: ok ({len(env.datasets)} datasets)")
env.close()
PYCHECK
```

На macOS/Windows удобнее запускать проект в Linux VM/container или вручную
поставить совместимые LLVM tools и Python-зависимости. Ключевое требование для
экспериментов: команды `opt`, `llc`, `llvm-size`, `llvm-dis`, `llvm-extract`,
`llvm-as` должны быть доступны в `PATH`.

## Запуск

```bash
llvm-ir-build-dataset --max-benchmarks 3 --overwrite
```

Для запуска по сохранённому набору benchmark-ов:

```bash
llvm-ir-build-dataset \
  --benchmark-file ../diplom_LLVM_IR/experiments/benchmark_sets/autotune_stratified_30.csv \
  --output-dir ./datasets/autotune_stratified_30_functions_bc \
  --work-dir ./build_workspace/autotune_stratified_30 \
  --overwrite
```

Эквивалентно через модуль:

```bash
python -m llvm_ir.stages.dataset.builder --max-benchmarks 3 --overwrite
```

Основные параметры:

- `--dataset` - имя датасета CompilerGym, по умолчанию `cbench-v1`;
- `--benchmark-file` - CSV с колонкой `benchmark_uri`; если задан, `--dataset` не используется;
- `--output-dir` - каталог с финальными per-function `.bc`;
- `--work-dir` - каталог промежуточных `.bc`/`.ll`;
- `--top-percent` - доля самых больших функций, по умолчанию `20.0`;
- `--no-function-selection` - сохранить все функции без top-percent отбора;
- `--max-benchmarks` - лимит для smoke-прогона;
- `--overwrite` - очистить выходной каталог перед запуском;
- `--keep-intermediate` - оставить промежуточные файлы.

## Тесты

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Если пакет установлен в editable-режиме, можно запускать и обычный `pytest`.

Тесты разложены по частям пайплайна:

- `tests/test_dataset_builder.py` - построение датасета и отбор функций;
- `tests/test_pass_search.py` - per-function поиск pass sequence, CEM и Random Search;
- `tests/test_translation_unit.py` - pass-order graph, TU-эвристики и whole-TU evaluation;
- `tests/aggregation/test_aggregation.py` - расширенные эвристики агрегации;
- `tests/test_pipeline_integration.py` - общий smoke-тест связки stage 1 -> stage 2.

## Сравнение Подбора Pass-Ов

Для быстрого CEM baseline на per-function `.bc`:

```bash
PYTHONPATH=src python3 -m llvm_ir.stages.function_search.pass_search \
  --dataset-dir datasets/autotune_stratified_30_functions_bc \
  --algorithm cem \
  --limit 30 \
  --steps 6 \
  --iterations 3 \
  --candidates 8
```

Скрипт пишет `comparison.json` и `comparison.csv` в
`experiments/pass_search_compare/<timestamp>/`.

CEM вынесен в `src/llvm_ir/stages/function_search/cem.py`: это алгоритм поиска
последовательности pass-ов для одной функции.
Random Search вынесен в `src/llvm_ir/stages/function_search/random_search.py`:
он равномерно сэмплирует последовательности pass-ов без обучения вероятностного
распределения и полезен как простой baseline для CEM.
`src/llvm_ir/stages/function_search/pass_search.py` оставляет на себе
LLVM-обвязку, замер `.text` и CLI, поэтому рядом можно добавлять другие
алгоритмы поиска с таким же per-function интерфейсом.

По умолчанию CEM использует STOP-action и best-prefix evaluation: кандидат может
закончить цепочку раньше `--steps`, а результатом считается лучший измеренный
prefix внутри цепочки. Для сравнения со старым fixed-length режимом можно
добавить `--no-stop-action`.

По умолчанию CEM оценивает ровно `--iterations * --candidates` сэмплов на
функцию. Дополнительная оценка всех уникальных циклических сдвигов отключена,
потому что резко увеличивает число реальных LLVM-замеров. Старый расширенный
режим можно включить флагом `--sequence-shifts`; тогда максимум замеров на
функцию становится `--iterations * --candidates * --steps`.

Для запуска Random Search вместо CEM:

```bash
PYTHONPATH=src python3 -m llvm_ir.stages.function_search.pass_search \
  --dataset-dir datasets/autotune_stratified_30_functions_bc \
  --algorithm random \
  --limit 30 \
  --steps 6 \
  --iterations 3 \
  --candidates 8 \
  --jobs 8
```

`--jobs` включает параллельную обработку функций. Для длинных full-dataset
запусков это заметно быстрее, а строки в итоговом JSON всё равно сохраняются в
детерминированном порядке.

PPO-метод из `llvm-minimizer` сравнивается с тем же датасетом после обучения или
при наличии checkpoint-а. В `llvm-minimizer` один входной `.bc` считается одним
эпизодом, поэтому per-function датасет подходит напрямую.

## Графы Порядка Pass-Ов

После запуска per-function поиска можно построить графы для stage 3:

```bash
PYTHONPATH=src python3 -m llvm_ir.stages.translation_unit.graph.order_graph \
  --input experiments/pass_search_compare/cem_shifts_all_seed7/comparison.json \
  --algorithm cem \
  --weight-mode count \
  --output experiments/translation_unit_graphs/cem_shifts_all_seed7/order_graphs.json
```

Для каждого benchmark-а выходной JSON содержит:

- `nodes` - pass-ы, встретившиеся в найденных последовательностях;
- `edges` - ориентированные отношения порядка `source -> target`;
- `weight` - сила поддержки отношения.

Режимы веса:

- `--weight-mode count` - каждое function-level подтверждение ребра добавляет `1`;
- `--weight-mode delta` - каждое подтверждение добавляет выигрыш функции
  `baseline_size - best_size`, поэтому последовательности с большим выигрышем
  сильнее влияют на граф;
- `--weight-mode count_distance` и `--weight-mode delta_distance` - используют
  ту же базовую поддержку, но усиливают близкие pass-ы: вклад пары умножается
  на `ceil(12 / distance)`, где `distance` - расстояние между pass-ами в
  function-level последовательности.

После построения графов можно запустить эвристики поиска общего пути pass-ов:

```bash
PYTHONPATH=src python3 -m llvm_ir.stages.translation_unit.path_heuristics \
  --input experiments/translation_unit_graphs/cem_shifts_all_seed7/order_graphs_delta.json \
  --heuristics all \
  --max-length 12 \
  --output experiments/translation_unit_heuristics/cem_shifts_all_seed7/delta/all_heuristics.json
```

Доступные эвристики:

- `greedy_consensus` - сортировка pass-ов по разнице исходящей и входящей поддержки;
- `dag_longest_path` - удаление конфликтных направлений и longest path в DAG;
- `cycle_breaking_max_path` - пока есть цикл, удаляет минимальное ребро цикла,
  затем делает topological sort и ищет maximum-weight path в полученном DAG;
- `exhaustive_len6` - перебирает все простые directed paths из 6 pass-ов и
  выбирает путь с максимальным graph score;
- `random_walk` - делает набор случайных блужданий по взвешенным рёбрам graph-а
  и выбирает лучший найденный путь по graph score;
- `beam_search` - beam search по путям с учётом поддержки и штрафом за конфликты;
- `weighted_toposort` - взвешенная топологическая сортировка; на DAG уважает
  входящие ограничения, а в циклах выбирает pass с лучшей разницей исходящей и
  входящей поддержки.

Эти графовые эвристики оцениваются двумя способами:

- proxy-метриками на pass-order graph (`order_score`, `conflict_score`,
  `net_score`, `adjacent_score`, `node_coverage`);
- реальным запуском найденной последовательности на цельном `.bc` benchmark-а.

Для whole-TU проверки:

```bash
PYTHONPATH=src python3 -m llvm_ir.stages.translation_unit.evaluate \
  --input experiments/translation_unit_heuristics/cem_shifts_all_seed7/delta/all_heuristics.json \
  --site-data ../diplom_LLVM_IR/.compiler_gym/site_data \
  --bitcode-dir experiments/translation_unit_bitcode/autotune_stratified_30 \
  --output experiments/translation_unit_eval/cem_shifts_all_seed7/delta/tu_eval_all_heuristics.json
```

Evaluation применяет pass-ы по одному, измеряет `.text` после каждого шага и
сохраняет как финальный размер всей последовательности, так и лучший префикс.

Для top-k TU-эвристик можно дополнительно считать не только процент выигрыша по
размеру `.text`, но и процент по количеству машинных инструкций. Для этого
используется флаг `--measure-instructions`: evaluator после каждого префикса
собирает объектный файл через `llc`, дизассемблирует его через `llvm-objdump -d`
и считает строки инструкций.

```bash
PYTHONPATH=src python3 -m llvm_ir.stages.translation_unit.evaluate_topk_paths \
  --graph runs/full_random_heuristic_20260602_231430/random/translation_unit_graphs/order_graphs_delta_with_starts.json \
  --bitcode-dir experiments/translation_unit_bitcode/autotune_stratified_30 \
  --output runs/cycle_top10_starts_top10_paths_instr/tu_eval.json \
  --heuristic cycle_breaking_top_starts_top_paths \
  --top-starts 10 \
  --paths-per-start 10 \
  --max-length 12 \
  --min-edge-weight 1 \
  --measure-instructions
```

В `summary` при таком запуске появляются дополнительные поля:

- `weighted_best_percent` - взвешенный процент выигрыша по размеру `.text`;
- `weighted_best_instruction_percent` - взвешенный процент выигрыша по числу
  машинных инструкций;
- `total_best_delta` - суммарный выигрыш по `.text`;
- `total_best_instruction_delta` - суммарный выигрыш по количеству машинных
  инструкций;
- `beats_oz_best_instruction` - сколько benchmark-ов лучше `-Oz` по числу
  машинных инструкций.

## Расширенные Эвристики Агрегации

Пакет `src/llvm_ir/heuristics/aggregation/` строит общий `PassGraph` по
per-function результатам и сравнивает набор эвристик агрегации. Основная цель -
получить TU-level candidates, которые лучше учитывают конфликтующие локальные
порядки pass-ов, чем простые graph heuristics.

Доступные эвристики:

- `wfas_eades` - weighted feedback-arc Eades ordering;
- `scc_ordering` - condensation graph по SCC и ordering внутри компонент;
- `pagerank` - PageRank на прямом и обратном графах предпочтений;
- `position_median` - weighted median нормализованной позиции pass-а;
- `hpp` - harmful-pass pruning перед построением порядка;
- `beam_diversity` - top-k beam search с diversity-штрафом;
- `cluster_aware` - группировка функций и слияние рангов по кластерам;
- `ilp_arrangement` - optional PuLP ILP/LP arrangement с fallback;
- `markov_hitting` - Markov stationary/hitting-style ordering;
- `voting_ensemble` - Borda aggregation нескольких эвристик;
- `hpp_eades_topk` - основной комбинированный вариант HPP + Eades + diverse top-k.

Graph-only сравнение всех aggregation-эвристик:

```bash
PYTHONPATH=src python3 -m llvm_ir.scripts.run_aggregation \
  --dataset experiments/pass_search_compare/cem_shifts_all_seed7/comparison.json \
  --algorithm cem \
  --all \
  --compare \
  --no-tu-eval \
  --output experiments/aggregation_graph_only/cem_shifts_all_seed7
```

Экспорт путей aggregation-эвристик в формат, совместимый с whole-TU evaluation:

```bash
PYTHONPATH=src python3 -m llvm_ir.heuristics.aggregation.export_paths \
  --input experiments/pass_search_compare/cem_shifts_all_seed7/comparison.json \
  --algorithm cem \
  --heuristics all \
  --max-length 12 \
  --beam-width 16 \
  --top-k 4 \
  --output experiments/aggregation_heuristics/cem_shifts_all_seed7/all_aggregation_heuristics.json
```

После этого файл можно передать в `llvm_ir.stages.translation_unit.evaluate` так
же, как результат базовых TU-эвристик.


## Материал Для Текста Диплома

Этот раздел собран как краткий технический конспект для написания диплома по
проекту. Его можно использовать как исходный материал для введения, описания
метода, экспериментальной части и обсуждения результатов.

### Цель Работы

Цель работы - исследовать, можно ли автоматически подбирать последовательности
LLVM optimization pass-ов так, чтобы получать меньший машинный код, чем при
использовании стандартной оптимизации `-Oz`, и переносить локально найденные
per-function последовательности на уровень всей единицы трансляции.

В LLVM порядок pass-ов существенно влияет на итоговый код: один pass может
создать возможности для другого pass-а, но также может уничтожить структуру,
которую другой pass мог бы оптимизировать. Поэтому задача выбора pass sequence
является комбинаторной: пространство последовательностей быстро растёт с
увеличением длины цепочки и числа доступных pass-ов. В работе эта задача
разделена на два уровня:

1. `function-level search` - для каждой отдельной функции ищется хорошая
   последовательность pass-ов прямым запуском LLVM и измерением результата;
2. `translation-unit aggregation` - по найденным function-level
   последовательностям строится граф предпочтений порядка pass-ов, после чего
   графовые эвристики предлагают одну общую последовательность для whole-TU
   bitcode.

### Данные И Объекты Измерения

Основной экспериментальный набор состоит из `30` benchmark-ов / translation
units и `444` выбранных функций. Функции получены из LLVM bitcode benchmark-ов,
а затем используются для per-function поиска. Whole-TU проверка выполняется на
полных `.bc` файлах benchmark-ов из
`experiments/translation_unit_bitcode/autotune_stratified_30`.

В проекте используются два типа измерений:

- размер секции `.text`, измеряемый после компиляции bitcode в object file;
- количество машинных инструкций, измеряемое как число строк инструкций в
  `llvm-objdump -d` после `llc -filetype=obj`.

Для whole-TU последовательностей pass-ы применяются по одному. После каждого
префикса измеряется результат, поэтому в отчётах есть две величины:

- `final` - результат после применения всей найденной последовательности;
- `best prefix` - лучший результат среди всех префиксов этой последовательности.

В дипломе лучше использовать `best prefix` как основную метрику, потому что она
соответствует реальному сценарию early stopping: если дальнейшие pass-ы портят
результат, можно остановиться на лучшем уже измеренном префиксе.

### Function-Level Поиск

На уровне отдельных функций реализованы два основных метода:

- `CEM` - Cross-Entropy Method, который обновляет вероятностное распределение по
  pass-ам на основе elite-кандидатов;
- `Random Search` - простой baseline, равномерно сэмплирующий цепочки pass-ов.

В текущей серии экспериментов лучшим источником данных для построения TU-графов
стал `Random Search`. Его полный результат находится в
`runs/full_random_heuristic_20260602_231430/random/pass_search/comparison.json`.

Результаты `Random Search` на `444` функциях:

| Метрика | Значение |
|---|---:|
| Функций | 444 |
| Улучшены по `.text` | 334 / 444 = 75.23% |
| Total `.text` delta | 38521 байт |
| Mean `.text` delta | 86.76 байт на функцию |
| `Random Search` лучше `-Oz` по `.text` | 208 / 444 = 46.85% |
| Total `.text` delta для `-Oz` | 14271 байт |
| Отношение total delta Random / `-Oz` | примерно 2.70x |

Результаты по количеству машинных инструкций находятся в
`runs/full_random_heuristic_20260602_231430/random/pass_search/instruction_count_comparison.json`:

| Метрика | Значение |
|---|---:|
| Baseline instructions | 119024 |
| Random Search instructions | 110871 |
| Total instruction delta | 8153 |
| Weighted instruction improvement | 6.85% |
| Улучшены по инструкциям | 298 / 444 = 67.12% |
| `Random Search` лучше `-Oz` по инструкциям | 169 / 444 = 38.06% |
| Total instruction delta для `-Oz` | 3603 |
| Отношение instruction delta Random / `-Oz` | 2.26x |

Вывод для диплома: даже простой Random Search даёт заметный function-level
выигрыш и является сильным источником локальных последовательностей pass-ов для
последующей агрегации. При этом он не требует обучения модели и поэтому удобен
как воспроизводимый baseline.

### Построение Pass-Order Graph

Для каждого benchmark-а строится ориентированный взвешенный граф порядка
pass-ов. Вершина графа - LLVM pass. Ребро `p_i -> p_j` означает, что в одной из
хороших function-level последовательностей pass `p_i` стоял раньше pass `p_j`.

Если функция имеет найденную последовательность `a, b, c`, то она добавляет в
граф не только соседние отношения, но все pairwise-отношения:

```text
a -> b
a -> c
b -> c
```

Базовые режимы весов:

- `count`: каждая функция добавляет `+1` к каждому подтверждённому ребру;
- `delta`: функция добавляет `max(baseline_size - best_size, 0)`, то есть
  функции с большим выигрышем сильнее влияют на граф.

Дополнительно добавлены distance-aware режимы:

- `count_distance`;
- `delta_distance`.

В этих режимах близкие pass-ы получают больший вклад. Формула для пары pass-ов:

```text
edge_weight += base_support * ceil(12 / distance)
```

где `distance` - расстояние между pass-ами в function-level последовательности.
Например, для последовательности `a, b, c, d` и `base_support = 2`:

```text
a -> b: 2 * ceil(12 / 1) = 24
a -> c: 2 * ceil(12 / 2) = 12
a -> d: 2 * ceil(12 / 3) = 8
```

Смысл этой модификации: соседние pass-ы отражают более сильную локальную
зависимость, чем просто факт, что один pass встретился где-то раньше другого.
Поэтому distance-aware граф лучше кодирует локальный порядок из найденных
function-level решений.

### TU-Level Эвристики

После построения графа задача состоит в том, чтобы выбрать одну общую
последовательность pass-ов для всего benchmark-а. В проекте реализованы базовые
и top-k эвристики:

- `greedy_consensus` - сортирует pass-ы по разнице исходящей и входящей
  поддержки;
- `dag_longest_path` - строит DAG из предпочтений и ищет длинный/тяжёлый путь;
- `cycle_breaking_max_path` - удаляет минимальные рёбра из циклов, затем ищет
  maximum-weight path в полученном DAG;
- `random_walk` - сэмплирует случайные блуждания по графу и выбирает лучшие;
- `exhaustive_len6` - перебирает все простые пути фиксированной длины 6;
- `beam_search` - строит путь постепенно, удерживая ограниченное число лучших
  кандидатов;
- `cycle_breaking_top_starts_top_paths` - выбирает несколько стартовых pass-ов
  по `start_counts`, а затем для каждого старта ищет несколько лучших путей.

Основной вариант, который показал лучшие результаты в текущих экспериментах, -
`cycle_breaking_top_starts_top_paths`. Он работает так:

1. По function-level random search выбираются наиболее сильные стартовые pass-ы
   (`start_counts`).
2. В графе циклы удаляются итеративно: пока цикл существует, из найденного цикла
   удаляется ребро минимального веса.
3. После удаления циклов получается DAG.
4. В DAG для выбранных стартовых вершин ищутся top-k путей с максимальным весом,
   ограниченные `max_length`.
5. Каждый candidate path проверяется реальным запуском на whole-TU bitcode.
6. Для каждого benchmark-а выбирается кандидат с лучшим реальным `.text`
   результатом; дополнительно сохраняется лучший префикс по машинным
   инструкциям.

Важно: графовый score используется только для генерации кандидатов. Финальный
выбор среди top-k путей делается прямым запуском LLVM на whole-TU bitcode.

#### `cycle_breaking_superpath_topk`

`cycle_breaking_superpath_topk` развивает эту идею в два этапа. Сначала
`cycle_breaking_top_starts_top_paths` генерирует короткие сегменты, затем каждый
сегмент реально измеряется на whole TU через общий prefix cache. После этого
строится граф супер-вершин: одна супер-вершина равна лучшему измеренному
префиксу короткой последовательности pass-ов, а её вес равен measured
`vertex_delta` этого префикса на TU. Если хвост исходного сегмента ухудшает
результат или падает, в супер-граф попадает только реально полезный префикс.
Ребро между двумя
сегментами существует, если в исходном order graph есть переход из последнего
pass-а первого сегмента в первый pass второго. Вес ребра сохраняется в отчёте как
`edge_score`, но после исправления единиц измерения не участвует в основном
score, а используется только как tie-breaker. Основной score суперпути - сумма
измеренных `vertex_delta` его сегментов.

Перебор суперпутей ограничен, чтобы не раздувать память на плотных графах:

- `--superpath-beam-factor` задаёт ширину послойного beam как
  `beam_factor * top_k` (default `5`);
- `--superpath-max-candidates` задаёт глобальный лимит сгенерированных
  частичных цепочек (default `100000`); при срабатывании лимита в строке отчёта
  появляется `superpath_truncated: true`;
- `--segment-top-k` по умолчанию `100`;
- `--segment-min-length` должен быть не меньше `2`;
- `--superpath-min-segment-delta` отбрасывает сегменты без достаточного
  measured best-prefix delta (default `1`);
- `--segment-max-jaccard` ограничивает похожесть сегментов перед замером
  (default `0.75`);
- `--superpath-max-overlap` ограничивает multiset-пересечение pass-ов между
  склеиваемыми сегментами (default `1`).

На маленьких или разреженных графах действует graceful degradation. Сначала
используются сегменты настроенной длины, затем нижняя граница опускается до `2`,
а потом до `1`. Для графов не больше `--tiny-graph-threshold` вершин (default
`4`) перебираются все простые пути, включая одиночные вершины. Если измеренные
пути существуют, но delta-фильтр удалил их все, evaluator оставляет top-3 по
`vertex_delta` и выполняет их как безопасные fallback-кандидаты. Поэтому
`no superpath candidates` остаётся только для пустого графа или случая, когда все
сырые пути завершились `full_failure`.

Новые диагностические поля строки отчёта:

| Поле | Значение |
|---|---|
| `segment_length_floor` | Фактическая нижняя граница длины сегмента: настроенная, `2` или `1` |
| `tiny_graph_mode` | Использован полный перебор простых путей tiny-графа |
| `segments_truncated_to_best_prefix` | Число сегментов, усечённых до полезного префикса |
| `segment_delta_filter_bypassed` | Пул восстановлен top-3 сегментами без delta-фильтра |

При `--measure-instructions` по умолчанию используется
`--instruction-measurement deferred`: все кандидаты сначала сравниваются только
по размеру, затем machine instruction count измеряется лишь для префиксов
выбранного по размеру кандидата. Baseline и `-Oz` измеряются сразу. Один object
file используется одновременно для `llvm-size` и `llvm-objdump`, поэтому eager
режим тоже не запускает `llc` дважды. `--instruction-measurement eager` оставлен
для полной диагностики instruction count каждого candidate row. Поле
`instruction_eval_cost` показывает число дополнительных instruction-компиляций
в deferred-фазе; instruction-поля неизбранных candidate rows остаются `null`.

Бюджет реальных запусков отражается в отчёте: `segment_eval_cost` показывает
число cache miss на фазе измерения сегментов, `superpath_eval_cost` - число
cache miss на фазе оценки склеек. Параметр `--superpath-eval-top-k` позволяет
оценивать больше склеек, чем основной `--top-k`; значение `0` означает
использовать `--top-k`. Это полезно, потому что сегменты уже оплачены prefix
cache-ом, и оценка склейки часто добавляет только хвостовые pass-ы.


#### `chunk_forest`

`chunk_forest` строит кандидаты не из отдельных рёбер order graph, а из чанков -
частых смежных подпоследовательностей pass-ов, найденных в per-function
результатах search-а. Для каждого benchmark-а evaluator майнит n-граммы,
замыкает их влево/вправо, добавляет несколько macro-чанков из лучших целых
function-level последовательностей и одиночные pass-ы как fallback для маленьких
benchmark-ов. Затем строится граф чанков: стартовые веса показывают, какие чанки
часто стоят в начале хороших function-level решений, а рёбра показывают, какие
чанки встречаются близко друг за другом. Из этого графа сэмплируется большой пул
candidate path-ов, после чего жадный prefix-forest selection выбирает пути с
учётом ценности чанков, разнообразия и числа новых prefix-cache узлов. По
умолчанию используется две волны: первая волна измеряется на whole TU, затем
ценность чанков пересчитывается по реальным TU-маргиналам в байтах, а вторая
волна выбирается уже по этой откалиброванной шкале.

Основные параметры evaluator-а:

```bash
PYTHONPATH=src python3 -m llvm_ir.stages.translation_unit.evaluate_chunk_forest \
  --comparison runs/<run>/random/pass_search/comparison.json \
  --algorithm random \
  --bitcode-dir experiments/translation_unit_bitcode/autotune_stratified_30 \
  --output runs/<run>/random/translation_unit_chunk_forest/tu_eval_chunk_forest.json \
  --paths 500 \
  --waves 2 \
  --pool-size 100000 \
  --walk-seed 7 \
  --ngram-max 4 \
  --closure-theta 0.8 \
  --min-support 2 \
  --top-chunks 30 \
  --macro-top 3 \
  --max-length 12 \
  --lambda-cache 0.0 \
  --gamma-diversity 0.5 \
  --max-real-evals-per-benchmark 0
```

`--lambda-cache 0.0` включает автоматическую калибровку штрафа за новый узел
prefix forest: медиана ценности чанка делится на среднюю длину чанка, чтобы один
cache miss был соизмерим с одним pass-ом ценности. Во второй волне measured
TU-байты не складываются напрямую с function-level весами: немеренные чанки
получают `mined_weight * scale`, где `scale` вычисляется только по чанкам,
которые уже получили TU-маргиналы. В отчёте для каждой выбранной строки есть
`chunks_mined`, `chunks_macro`, `chunks_single`, `pool_unique_paths`,
`wave1_real_evals`, `wave2_real_evals`, `chunks_remeasured`, `rescoring_scale`,
`top_chunks` и общий `real_evals`.

Для nightly-прогона Chunk-Forest включается отдельно:

```bash
RUN_CHUNK_FOREST=1 \
CF_PATHS=500 \
CF_WAVES=2 \
CF_POOL=100000 \
CF_MAX_EVALS=0 \
RUN_CEM=0 RUN_RANDOM=1 \
scripts/nightly_big_pass_search_and_heuristics.sh
```

Предписанная матрица сравнения для диплома: `chunk_forest` на `500` путей в `2`
волны, `chunk_forest` на `500` путей в `1` волну, `random_walk_top500` и
`random_walk_top1000`. Сравнивать их нужно не только по числу candidate paths, но
и по `real_evals`, потому что общий prefix cache делает стоимость путей с общими
префиксами существенно ниже.

### Основные TU Результаты

Ниже приведены лучшие сохранённые отчёты на `30` benchmark-ах. Основная
метрика - weighted best improvement: суммарный выигрыш, делённый на суммарный
baseline, поэтому большие benchmark-и влияют на итог пропорционально своему
размеру. Все значения взяты из JSON-отчётов в `runs/`; `n/a` означает, что в
этом запуске machine instruction count не измерялся.

#### Лучшие Graph/Top-K/TU Эвристики

| Эвристика / запуск | Report | `.text` total best delta | `.text` weighted best | Instruction total best delta | Instruction weighted best | Better than `-Oz` |
|---|---|---:|---:|---:|---:|---:|
| `random_walk_top1000` | `runs/random_walk_top1000_instr_20260608_224901/tu_eval_random_walk_top1000_instr.json` | 77495 | 7.5013% | 17383 | 8.4306% | 21/30 |
| `cycle_breaking_top_starts_top_paths` (`top20 x top20`) | `runs/cycle_top20_starts_top20_paths_instr_20260605_114124/tu_eval_cycle_top20_starts_top20_paths_instr.json` | 74331 | 7.1950% | 16913 | 8.2026% | 20/30 |
| `cycle_breaking_top_starts_top_paths` (`top10 x top10`, `delta_distance`) | `runs/cycle_top10_starts_top10_paths_delta_distance_instr_20260607_101117/tu_eval_cycle_top20_starts_top10_paths_delta_distance_instr.json` | 74055 | 7.1683% | 16843 | 8.1687% | 20/30 |
| `cycle_breaking_max_path_top1000` | `runs/cycle_max_path_top1000_20260604_211303/tu_eval_cycle_breaking_max_path_top1000.json` | 73294 | 7.0946% | n/a | n/a | 20/30 |
| `cycle_breaking_diverse_starts_top10` | `runs/full_random_heuristic_20260602_231430/random/translation_unit_eval/delta/tu_eval_top10_cycle_breaking_diverse_starts.json` | 73158 | 7.0815% | n/a | n/a | 20/30 |
| `cycle_breaking_superpath_topk` | `runs/superpath_top250_len20_segmin1_full_20260612_011504/tu_eval_cycle_superpath_top250_instr.json` | 66982 | 6.4836% | 15691 | 7.6100% | 22/30 |
| `cycle_breaking_max_path_top10` | `runs/full_random_heuristic_20260602_231430/random/translation_unit_eval/delta/tu_eval_top10_cycle_breaking_max_path.json` | 65449 | 6.3353% | n/a | n/a | 19/30 |
| `chunk_forest` (`500 paths`, `2 waves`) | `runs/post_bugfix_random_20260611_120635/random/translation_unit_chunk_forest/tu_eval_chunk_forest.json` | 63506 | 6.1472% | 14863 | 7.2084% | 23/30 |
| `beam_search` | `runs/post_bugfix_random_20260611_120635/random/translation_unit_eval/delta/tu_eval_all_heuristics.json` | 62814 | 6.0802% | n/a | n/a | 23/30 |
| `cycle_breaking_max_path` | `runs/full_random_heuristic_20260602_231430/random/translation_unit_eval/delta/tu_eval_all_heuristics.json` | 62745 | 6.0735% | n/a | n/a | 19/30 |
| `random_walk_top10` | `runs/full_random_heuristic_20260602_231430/random/translation_unit_eval/delta/tu_eval_top10_random_walk_exhaustive_len6.json` | 62060 | 6.0072% | n/a | n/a | 20/30 |
| `exhaustive_len6` | `runs/post_bugfix_random_20260611_120635/random/translation_unit_eval/delta/tu_eval_all_heuristics.json` | 58989 | 5.7099% | n/a | n/a | 21/30 |
| `random_walk` | `runs/post_bugfix_random_20260611_120635/random/translation_unit_eval/delta/tu_eval_all_heuristics.json` | 58946 | 5.7058% | n/a | n/a | 22/30 |
| `weighted_toposort` | `runs/post_bugfix_full_20260611_091641/cem/translation_unit_eval/delta/tu_eval_all_heuristics.json` | 54708 | 5.2956% | n/a | n/a | 20/30 |
| `dag_longest_path` | `runs/full_random_heuristic_20260602_231430/random/translation_unit_eval/delta/tu_eval_all_heuristics.json` | 50846 | 4.9217% | n/a | n/a | 18/30 |
| `greedy_consensus` | `runs/full_random_heuristic_20260602_231430/random/translation_unit_eval/delta/tu_eval_all_heuristics.json` | 34300 | 3.3201% | n/a | n/a | 18/30 |

#### Лучшие Aggregation Эвристики

Все строки ниже взяты из одного full-30 запуска:
`runs/aggregation_all11_random_full_20260612_011206/tu_eval_all_aggregation_heuristics.json`.
Для top-k aggregation-методов указана лучшая full-30 строка `top1`, потому что
последующие top-k строки в этом отчёте покрывают не все `30` benchmark-ов.

| Aggregation эвристика | `.text` total best delta | `.text` weighted best | Better than `-Oz` | Failed |
|---|---:|---:|---:|---:|
| `hpp_eades_topk_top1` | 62793 | 6.0782% | 24/30 | 3 |
| `markov_hitting` | 60829 | 5.8881% | 21/30 | 2 |
| `beam_diversity_top1` | 59670 | 5.7759% | 22/30 | 2 |
| `scc_ordering` | 56212 | 5.4411% | 22/30 | 0 |
| `pagerank` | 25095 | 2.4291% | 22/30 | 1 |
| `hpp` | 24412 | 2.3630% | 20/30 | 1 |
| `ilp_arrangement` | 24412 | 2.3630% | 20/30 | 1 |
| `wfas_eades` | 24412 | 2.3630% | 20/30 | 1 |
| `voting_ensemble` | 24158 | 2.3384% | 22/30 | 1 |
| `position_median` | 21222 | 2.0542% | 20/30 | 1 |
| `cluster_aware_top1` | 20960 | 2.0289% | 20/30 | 2 |

Выводы из таблиц:

- Лучший сохранённый результат остаётся у `random_walk_top1000`: `7.5013%` по
  `.text` и `8.4306%` по числу машинных инструкций.
- Лучшая детерминированная graph/top-k схема -
  `cycle_breaking_top_starts_top_paths` (`top20 x top20`): `7.1950%` по `.text`
  и `8.2026%` по инструкциям.
- `cycle_breaking_superpath_topk` после фиксов единиц измерения даёт `6.4836%`
  по `.text`; лучший более поздний fixed-запуск был стабильнее по реализации,
  но ниже по сохранённым метрикам (`6.3522%`).
- `chunk_forest` в первом full-запуске даёт `6.1472%` по `.text` и `7.2084%` по
  инструкциям, но средний пул получился маленьким (`~20` уникальных путей на
  benchmark), поэтому вторая волна фактически не раскрылась.
- Среди aggregation-эвристик сильнее всего выглядит `hpp_eades_topk_top1`
  (`6.0782%`), затем `markov_hitting` (`5.8881%`) и `beam_diversity_top1`
  (`5.7759%`).

### Интерпретация Для Диплома

Главный результат работы состоит в том, что локальные function-level находки
можно агрегировать в полезные whole-TU последовательности. Прямая оптимизация
каждой функции даёт сильный сигнал о том, какие pass-ы и какие относительные
порядки полезны. Pass-order graph превращает набор локальных решений в
структуру, пригодную для поиска общего порядка pass-ов.

При этом графовый score сам по себе недостаточен: путь с большим весом в графе
не гарантирует лучший машинный код на whole-TU. Поэтому в финальной схеме graph
heuristic используется как генератор небольшого множества кандидатов, а качество
каждого кандидата проверяется реальным запуском LLVM. Такой hybrid-подход
уменьшает пространство поиска, но сохраняет связь с настоящей целевой метрикой.

Distance-aware веса являются попыткой точнее кодировать локальные зависимости
между pass-ами. Они дали небольшой положительный эффект для `top10 x top10`, но
не превзошли более широкий поиск `top20 x top20`. Это можно интерпретировать
так: локальная близость действительно несёт полезный сигнал, но разнообразие
кандидатов и прямой TU-evaluation остаются более важными факторами.

### Ограничения Экспериментов

При описании результатов важно явно указать ограничения:

- используется ограниченный набор из `30` benchmark-ов;
- измерения зависят от версии LLVM tools и target backend;
- некоторые последовательности pass-ов могут приводить к падению `opt`; evaluator
  ловит такие ошибки и использует лучший уже найденный префикс;
- per-function оптимальный порядок не обязан быть оптимальным для whole-TU из-за
  межфункциональных эффектов и изменения контекста;
- top-k TU evaluation требует большого числа реальных запусков LLVM, поэтому
  масштабирование ограничено временем эксперимента.

### Что Можно Писать Как Вклад Работы

Возможная формулировка вклада:

1. Разработан воспроизводимый пайплайн для построения per-function LLVM bitcode
   датасета, поиска pass sequence и проверки результата на whole-TU bitcode.
2. Реализованы CEM и Random Search для function-level подбора pass-ов с
   best-prefix evaluation.
3. Предложена графовая агрегация function-level последовательностей в
   pass-order graph с несколькими режимами взвешивания, включая `delta` и
   `delta_distance`.
4. Реализована эвристика `cycle_breaking_top_starts_top_paths`, которая удаляет
   циклы, выбирает сильные стартовые pass-ы и генерирует top-k путей в DAG.
5. Добавлена real-TU проверка candidate path-ов с prefix-cache и измерением как
   `.text`, так и количества машинных инструкций.
6. Экспериментально показано, что предложенная схема даёт до `7.5013%`
   weighted improvement по `.text` и до `8.4306%` weighted improvement по числу
   машинных инструкций на наборе из `30` translation units.

### Где Лежат Основные Артефакты

| Артефакт | Путь |
|---|---|
| Function-level Random Search | `runs/full_random_heuristic_20260602_231430/random/pass_search/comparison.json` |
| Instruction-count для Random Search | `runs/full_random_heuristic_20260602_231430/random/pass_search/instruction_count_comparison.json` |
| Top10 x Top10 TU eval | `runs/cycle_top10_starts_top10_paths_instr/tu_eval.json` |
| Top20 x Top20 TU eval | `runs/cycle_top20_starts_top20_paths_instr_20260605_114124/tu_eval_cycle_top20_starts_top20_paths_instr.json` |
| Delta-distance Top10 x Top10 TU eval | `runs/cycle_top10_starts_top10_paths_delta_distance_instr_20260607_101117/tu_eval_cycle_top20_starts_top10_paths_delta_distance_instr.json` |
| Random Walk Top1000 TU eval | `runs/random_walk_top1000_instr_20260608_224901/tu_eval_random_walk_top1000_instr.json` |
| Top1000 cycle-breaking eval | `runs/cycle_max_path_top1000_20260604_211303/tu_eval_cycle_breaking_max_path_top1000.json` |
| Superpath Top250 len20 eval | `runs/superpath_top250_len20_segmin1_full_20260612_011504/tu_eval_cycle_superpath_top250_instr.json` |
| Chunk-Forest eval | `runs/post_bugfix_random_20260611_120635/random/translation_unit_chunk_forest/tu_eval_chunk_forest.json` |
| Aggregation all-11 TU eval | `runs/aggregation_all11_random_full_20260612_011206/tu_eval_all_aggregation_heuristics.json` |


## Полный Эксперимент

End-to-end прогон CEM и Random Search, построение графов, запуск базовых
TU-эвристик, aggregation-эвристик и whole-TU evaluation собраны в скрипт:

```bash
scripts/nightly_big_pass_search_and_heuristics.sh
```

Параметры задаются переменными окружения, например:

```bash
RUN_ID=full_seed7 \
LIMIT=0 \
STEPS=8 \
ITERATIONS=6 \
CANDIDATES=32 \
RANDOM_ITERATIONS=12 \
RANDOM_CANDIDATES=64 \
JOBS=8 \
scripts/nightly_big_pass_search_and_heuristics.sh
```

По умолчанию скрипт запускает оба алгоритма. Их можно разделять флагами:

```bash
RUN_ID=cem_seed7 RUN_CEM=1 RUN_RANDOM=0 \
scripts/nightly_big_pass_search_and_heuristics.sh

RUN_ID=random_seed7 RUN_CEM=0 RUN_RANDOM=1 \
scripts/nightly_big_pass_search_and_heuristics.sh
```

Скрипт пишет артефакты в `runs/<RUN_ID>/`, включая `summary.json` с основными
результатами выбранных алгоритмов.

## PPO Baseline

Starter-конфиг для обучения PPO:

```bash
cd ../llvm-minimizer
llvm-minimizer train --config ../LLVM_IR/configs/llvm_minimizer_ppo.yaml
```

Короткий воспроизводимый прогон для первого сравнения:

```bash
cd ../llvm-minimizer
MPLCONFIGDIR=/tmp/mpl .venv/bin/llvm-minimizer train \
  --config ../LLVM_IR/configs/llvm_minimizer_ppo_quick.yaml
```

Сравнение CEM с обученным checkpoint-ом:

```bash
cd ../LLVM_IR
MPLCONFIGDIR=/tmp/mpl \
PYTHONPATH=src:../llvm-minimizer/src \
../llvm-minimizer/.venv/bin/python -m llvm_ir.stages.function_search.pass_search \
  --dataset-dir datasets/autotune_stratified_30_functions_bc \
  --limit 20 \
  --steps 6 \
  --iterations 3 \
  --candidates 8 \
  --elite-size 3 \
  --seed 7 \
  --ppo-config configs/llvm_minimizer_ppo_quick.yaml \
  --ppo-checkpoint experiments/ppo_runs_quick/<run-id>/best.zip
```
