# LLVM_IR

Инструменты для построения датасета LLVM IR/bitcode на уровне отдельных
функций.

Первая часть пайплайна перенесена из прототипа `dataset-builder`: скрипт берёт
бенчмарки из `CompilerGym`, выгружает исходный `.bc`, режет его на функции,
считает размер функций в LLVM IR и сохраняет top N% самых больших функций как
отдельные `.bc` файлы.

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
python -m llvm_ir.dataset_builder --max-benchmarks 3 --overwrite
```

Основные параметры:

- `--dataset` - имя датасета CompilerGym, по умолчанию `cbench-v1`;
- `--benchmark-file` - CSV с колонкой `benchmark_uri`; если задан, `--dataset` не используется;
- `--output-dir` - каталог с финальными per-function `.bc`;
- `--work-dir` - каталог промежуточных `.bc`/`.ll`;
- `--top-percent` - доля самых больших функций, по умолчанию `20.0`;
- `--max-benchmarks` - лимит для smoke-прогона;
- `--overwrite` - очистить выходной каталог перед запуском;
- `--keep-intermediate` - оставить промежуточные файлы.

## Тесты

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Если пакет установлен в editable-режиме, можно запускать и обычный `pytest`.
