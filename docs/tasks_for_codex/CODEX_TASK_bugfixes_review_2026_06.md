# Задание для AI-агента (Codex): исправление багов по результатам код-ревью

## 1. Контекст

Репозиторий `LLVM_IR` — пайплайн из трёх стадий (dataset → function-level pass search → translation-unit aggregation/evaluation). По результатам код-ревью найдены 4 подтверждённых бага, несколько слабых мест дизайна и набор мелочей. Их нужно исправить, не ломая существующие интерфейсы CLI/JSON-отчётов и сохранив прохождение текущих 75 тестов (`PYTHONPATH=src python3 -m unittest discover -s tests -v`).

Общие правила:
- Не менять формат существующих JSON-отчётов (ключи, структуру), кроме случаев, где это явно указано.
- На каждый исправляемый баг добавить регрессионный unit-тест, который падает до фикса и проходит после.
- Стиль кода — как в репозитории: dataclasses, type hints, без новых обязательных зависимостей.
- После всех правок прогнать полный тест-сьют и убедиться, что он зелёный.

---

## 2. Блок A — подтверждённые баги (обязательно)

### A1. Инвертированный порядок в pagerank-агрегации

Файл: `src/llvm_ir/heuristics/aggregation/pagerank_ordering.py`.

Воспроизведение: датасет из 5 функций, у всех `sequence = ["a", "b", "c"]`. `PageRankOrdering().aggregate(...)` возвращает `["c", "b", "a"]` — порядок перевёрнут.

Причина: ребро `p_i -> p_j` означает «i раньше j», поэтому высокий `incoming_rank` означает *позднюю* позицию pass-а. Текущая сортировка `key=lambda node: (-(incoming_rank[node] - outgoing_rank[node]), node)` ставит поздние pass-ы в начало.

Что сделать:
1. Исправить знак: сортировать по убыванию `outgoing_rank[node] - incoming_rank[node]` (ранние pass-ы — первыми). Тай-брейк по имени ноды оставить.
2. Добавить тест в `tests/aggregation/test_aggregation.py`: для единогласного датасета `a -> b -> c` (минимум 3 функции, delta > 0) `chosen_sequence == ["a", "b", "c"]`.
3. Проверить, что `voting_ensemble` (использует pagerank как voter) после фикса всё ещё проходит свои тесты.

### A2. Лучший кандидат с размером 0 затирается худшим

Файлы: `src/llvm_ir/stages/function_search/cem.py` (функция `search_pass_sequence_for_function`) и `src/llvm_ir/stages/function_search/random_search.py` (функция `search_pass_sequence_randomly`).

Проблема: условие `elif best is None or result.size < (best.size or baseline_size + 1)`. Если `best.size == 0`, выражение `best.size or baseline_size + 1` из-за falsy-нуля даёт `baseline_size + 1`, и любой кандидат с размером меньше baseline заменяет идеальный нулевой результат.

Что сделать:
1. Заменить в обоих местах на явную проверку: best уже гарантированно имеет `size is not None` (он назначается только из успешных результатов), поэтому достаточно `result.size < best.size`. Если хочется сохранить защиту от None — `best.size if best.size is not None else baseline_size + 1`.
2. Тест (например, в `tests/test_pass_search.py`): фейковый `evaluate_candidate`, который первому кандидату возвращает `size=0`, второму `size=5`; убедиться, что в итоге `result.best.size == 0`. Покрыть и CEM, и random search.

### A3. Мёртвая ветка в dag_longest_path

Файл: `src/llvm_ir/stages/translation_unit/dag_longest_path.py`.

Проблема: во вложенном цикле `for target in order[left_index + 1:]` всегда `order_index[target] > order_index[source]`, поэтому ветка

```python
elif -net >= config.min_net_weight:
    if order_index[target] < order_index[source]:
        dag_edges[target].append((source, -net))
```

недостижима — сильные обратные предпочтения молча выбрасываются.

Что сделать:
1. Решение по семантике: эвристика должна оставаться DAG-овой относительно priority-порядка, поэтому **удалить мёртвый `elif` целиком** и добавить комментарий, что рёбра против priority-порядка сознательно игнорируются для сохранения ацикличности.
2. Тест: на графе с сильным обратным ребром (`b -> a` тяжелее, чем `a -> b`) проверить, что функция не падает и возвращает детерминированный путь; плюс простой тест на то, что поведение на DAG не изменилось (можно опереться на существующие тесты в `tests/test_translation_unit.py`).

### A4. Группировка по benchmark ломается при дефолтном неймировании датасета

Файлы: `src/llvm_ir/stages/dataset/builder.py` (функции `benchmark_short_name`, `build_dataset`) и `src/llvm_ir/stages/translation_unit/graph/order_graph.py` (`benchmark_id_from_function_name`), плюс `src/llvm_ir/stages/translation_unit/evaluate.py` (`benchmark_uri_from_id`).

Проблема: без `--benchmark-file` builder именует файлы `<benchmark>_<func>.bc` (без префикса датасета), а `benchmark_id_from_function_name` берёт первые два `_`-токена. В результате `qsort_main.bc` группируется в фиктивный benchmark «qsort_main», а `benchmark_uri_from_id` строит мусорный URI `benchmark://qsort/main`. Дополнительно: любой benchmark, в имени которого есть `_`, обрезается даже в режиме `--benchmark-file`.

Что сделать (минимально-инвазивный вариант):
1. В `build_dataset` всегда использовать `include_dataset_in_name=True` (убрать ветвление), чтобы имена файлов всегда были `<suite>_<benchmark>_<func>.bc`. Обновить docstring у `benchmark_short_name`.
2. Сделать разделение надёжным: в `benchmark_short_name` заменить `_` внутри *имени benchmark-а* на `-` перед склейкой (suite-имена CompilerGym уже не содержат `_`), т.е. после `sanitize_filename` дополнительно нормализовать сегмент имени benchmark-а. Тогда первые два `_`-токена всегда корректно восстанавливают `(suite, benchmark)`. Альтернатива, если она проще и не ломает существующий датасет `datasets/autotune_stratified_30_functions_bc`: использовать редкий разделитель `__` между suite/benchmark/function и научить `benchmark_id_from_function_name` сначала пробовать `__`, затем падать обратно на старый `_`-парсинг (для обратной совместимости с уже собранными данными).
3. ВАЖНО: существующие имена в `datasets/autotune_stratified_30_functions_bc` и старые отчёты в `runs/` должны продолжать парситься как раньше — добавить тест на текущие реальные имена (`chstone-v0_adpcm_decode.bc` -> `chstone-v0_adpcm`, `mibench-v1_lame-newmdct-1_<func>.bc` -> `mibench-v1_lame-newmdct-1`).
4. Тесты: `qsort_main.bc` больше не должен возникать как имя файла из builder-а (юнит-тест на `benchmark_short_name`); round-trip тест `benchmark_uri_from_id(benchmark_id_from_function_name(name))` для нескольких реалистичных URI, включая имя benchmark-а с `_`.

---

## 3. Блок B — слабые места дизайна (обязательно, но аккуратно)

### B1. Воспроизводимость: jobs=1 vs jobs>1

Файл: `src/llvm_ir/stages/function_search/pass_search.py`, функция `run_pass_search_jobs`.

Сейчас при `jobs <= 1` все функции делят один `random.Random(seed)`, а при `jobs > 1` каждая функция получает `seed + index * 1_000_003`. Один seed даёт разные результаты в зависимости от числа воркеров.

Что сделать: в последовательной ветке тоже создавать per-function RNG по той же формуле `random.Random(seed + index * 1_000_003)`. Вынести формулу в маленькую функцию `function_seed(seed, index)` и использовать в обеих ветках. Добавить тест: с фейковым алгоритмом, записывающим первые значения rng, результаты при jobs=1 совпадают с тем, что дала бы параллельная формула сидов.

### B2. filter_valid_passes валидирует только первый файл

Файл: `src/llvm_ir/stages/function_search/pass_search.py`.

Что сделать (без изменения поведения по умолчанию): добавить опцию `--validate-passes-on N` (по умолчанию 1 — текущее поведение). При N > 1 pass считается валидным, если он успешен хотя бы на одном из первых N файлов. В `config` отчёта записывать значение параметра. Тест с фейковым runner-ом: pass, падающий на первом файле, но валидный на втором, сохраняется при N=2 и фильтруется при N=1.

### B3. Кэш ошибок и рост диска в evaluate_topk_paths

Файл: `src/llvm_ir/stages/translation_unit/evaluate_topk_paths.py`, функция `evaluate_candidate_with_prefix_cache` и `evaluate_topk_for_benchmark`.

Что сделать:
1. В словарь-entry с ошибкой добавлять флаг и **не считать его терминальным навсегда**: достаточно пометить `"failed_pass": pass_name` и оставить текущее поведение break, но добавить в отчёт benchmark-а счётчик `prefix_failures` (число закэшированных ошибочных префиксов), чтобы падения были видны. Полноценный retry не нужен.
2. Диск: после обработки benchmark-а временная директория и так удаляется (`TemporaryDirectory`), но внутри одного benchmark-а файлы копятся. Добавить параметр CLI `--max-prefix-cache N` (0 = без лимита, по умолчанию 0): при превышении N закэшированных префиксов новые промежуточные `.bc` после замера удалять с диска и хранить в entry только `size`/`instruction_count` без `bc` (такой entry нельзя расширять — при попытке расширения пересчитывать от ближайшего предка с живым `bc`). Если это получается слишком сложно, допустимая упрощённая версия: при превышении лимита просто удалять `.o`-файлы (они после замера не нужны всегда) и документировать ограничение. Минимум: `.o`-файлы, создаваемые `measure_text_size` и `measure_machine_instruction_count`, удалять сразу после замера — это безусловная дешёвая победа, сделать обязательно.
3. Тест: после `measure_text_size` объектник не остаётся в workdir (либо счётчик файлов в workdir не растёт по `.o`).

### B4. Строка с error может быть выбрана как best и искажает summary

Файл: `src/llvm_ir/stages/translation_unit/evaluate_topk_paths.py`.

Что сделать: в выбранной строке (`selected_rows`) различать «кандидат упал целиком» и «упал хвост, best prefix валиден». Добавить в row поле `error_kind`: `""` | `"tail_failure"` (best_prefix_len > 0 при наличии error) | `"full_failure"`. В `summarize_evaluations` ничего не менять (совместимость), но в `add_instruction_summary`-подобном месте (или прямо в make_report_payload) добавить в summary heuristic-а счётчики `tail_failures` и `full_failures`. Тест на классификацию.

---

## 4. Блок C — мелочи (сделать, если не тянет за собой риски)

1. `order_graph.PassOrderGraph.add_sequence`: не инкрементировать `sequence_count`, если `support_weight <= 0` (или добавить отдельный счётчик `zero_weight_sequences`). Обновить затронутые тесты осознанно: если существующий тест фиксирует текущее поведение — выбрать вариант с отдельным счётчиком и не менять `sequence_count`.
2. `builder.safe_function_stem`: устранить молчаливые коллизии — `extract_functions_from_bc` должен при коллизии stem-ов добавлять короткий sha1-суффикс от исходного имени (как уже делается для длинных имён). Тест: `foo.bar` и `foo_bar` дают разные имена файлов.
3. `evaluate.evaluate_sequence_on_bitcode`: при ошибке посреди цепочки добавить в результат поле `applied_passes` (фактически применённый префикс), не меняя существующие поля. Обновить `evaluation_to_row`.
4. `scripts/nightly_big_pass_search_and_heuristics.sh`: убрать машинно-специфичный дефолт `SITE_DATA=/home/vladimir/...` — заменить на пустую строку и пропускать `--site-data`, если переменная пуста.
5. `_find_cycle` в `cycle_breaking_max_path.py` и Tarjan в `aggregation/graph_utils.py`: переписать на итеративный DFS (явный стек), поведение и порядок результатов не менять. Существующие тесты должны пройти без правок.
6. `count_llvm_ir_instructions` в builder: переименовывать не нужно, но дополнить docstring перечислением того, что считается («строки, не являющиеся метками/комментариями/декларациями, включая `}` и глобалы») — это эвристика отбора, а не точный счётчик.

---

## 5. Критерии приёмки

1. `PYTHONPATH=src python3 -m unittest discover -s tests -v` — зелёный, включая новые тесты.
2. Для каждого пункта A1–A4 есть регрессионный тест, падающий на коде до фикса.
3. Прогон smoke-цепочки не ломается (без CompilerGym): `python3 -m llvm_ir.stages.translation_unit.graph.order_graph` и `path_heuristics` на любом существующем comparison.json из тестовых фикстур.
4. Существующие JSON-ключи отчётов не удалены и не переименованы; новые поля только добавлены.
5. Старые имена файлов датасета (`datasets/autotune_stratified_30_functions_bc/*.bc`) парсятся в те же benchmark id, что и раньше (тест из A4.3).
6. В коммитах — по одному логическому изменению (A1, A2, ... отдельными коммитами), сообщения на английском в стиле `fix: invert pagerank ordering direction`.

## 6. Что НЕ делать

- Не трогать алгоритмическую логику CEM/эвристик сверх описанного (никаких «улучшений по пути»).
- Не добавлять обязательные зависимости (networkx, pulp остаются опциональными/неиспользуемыми).
- Не менять CLI-дефолты, кроме явно указанных (`--validate-passes-on`, `--max-prefix-cache` — новые опции с безопасными дефолтами).
- Не переписывать существующие отчёты в `runs/` и датасет в `datasets/`.
