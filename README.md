# LLVM_IR

Инструменты для построения датасета LLVM IR/bitcode на уровне отдельных
функций.

Проект строится как пайплайн из трёх частей:

1. `src/llvm_ir/stages/dataset/` - построение per-function датасета из
   CompilerGym и опциональный отбор top 20% функций по размеру LLVM IR.
2. `src/llvm_ir/stages/function_search/` - подбор оптимальной
   последовательности pass-ов для каждой функции. CEM сейчас подключён как
   первый алгоритм через общий интерфейс.
3. `src/llvm_ir/stages/translation_unit/` - контракт для будущих эвристик, которые будут
   собирать общую последовательность pass-ов на всю единицу трансляции из
   per-function результатов.

Первый шаг stage 3 - построение pass-order graph по найденным per-function
последовательностям. Для каждого benchmark-а строится ориентированный граф:
ребро `p_i -> p_j` означает, что в хотя бы одной лучшей последовательности
функции benchmark-а pass `p_i` встречался раньше pass `p_j`.

В корне пакета оставлены совместимые модули `dataset_builder.py`, `cem.py`,
`pass_search.py` и другие тонкие обёртки, чтобы старые импорты продолжали
работать.

## Установка для разработки

```bash
python -m pip install -e '.[dev]'
```

Для реального построения датасета дополнительно нужны:

- `compiler_gym`;
- LLVM CLI tools: `llvm-dis`, `llvm-extract`, `llvm-as`.

```bash
python -m pip install -e '.[compiler-gym]'
```

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
- `tests/test_pass_search.py` - per-function поиск pass sequence и CEM;
- `tests/test_translation_unit.py` - контракт будущего stage 3;
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
PYTHONPATH=src python3 -m llvm_ir.stages.translation_unit.order_graph \
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
- `beam_search` - beam search по путям с учётом поддержки и штрафом за конфликты.
- `weighted_toposort` - взвешенная топологическая сортировка; на DAG уважает
  входящие ограничения, а в циклах выбирает pass с лучшей разницей исходящей и
  входящей поддержки.

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
