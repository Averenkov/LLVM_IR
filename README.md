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
`src/llvm_ir/stages/function_search/pass_search.py` оставляет на себе
LLVM-обвязку, замер `.text` и CLI, поэтому рядом можно добавлять другие
алгоритмы поиска с таким же per-function интерфейсом.

По умолчанию CEM использует STOP-action и best-prefix evaluation: кандидат может
закончить цепочку раньше `--steps`, а результатом считается лучший измеренный
prefix внутри цепочки. Для сравнения со старым fixed-length режимом можно
добавить `--no-stop-action`.

PPO-метод из `llvm-minimizer` сравнивается с тем же датасетом после обучения или
при наличии checkpoint-а. В `llvm-minimizer` один входной `.bc` считается одним
эпизодом, поэтому per-function датасет подходит напрямую.

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
