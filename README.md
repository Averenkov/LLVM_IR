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

В корне пакета оставлены совместимые модули `dataset_builder.py`, `cem.py`,

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
  сильнее влияют на граф.

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
