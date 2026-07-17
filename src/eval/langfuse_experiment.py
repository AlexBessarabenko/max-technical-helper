"""Langfuse-эксперимент: датасет goldens + dataset run с Ragas-scores на трейсах.

Сценарий (паттерны HW5):
1. Идемпотентно создаёт датасет ``corporate_kb_goldens`` и наполняет его
   записями goldens.json без флага skip_rag (14 items).
2. Прогоняет каждый item через ``RAGPipeline.answer_traced`` с
   user_id="experiment" и session_id=run_name, затем привязывает трейсы
   к dataset run через низкоуровневый ``lf.api.dataset_run_items.create``.
3. Один раз прогоняет Ragas-оценку (``run_evaluation``) и пишет per-row
   метрики (faithfulness, answer_relevancy, context_recall) как scores
   на соответствующие трейсы через ``lf.create_score``.

Замечание по trace_id: ``lf.get_current_trace_id()`` после выхода из
root-observation возвращает None (проверено на SDK 4.14.0), а answer_traced
трейс наружу не возвращает. Поэтому trace_id восстанавливаются через
``lf.api.trace.list(session_id=run_name)`` с сопоставлением по
``trace.input["question"]`` — форма трейсов при этом не меняется
(никаких обёрточных спанов поверх max_request).

Запуск: ``.venv/bin/python -m src.eval.langfuse_experiment`` из корня проекта.
"""

import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from langfuse import Langfuse
from langfuse._client.datasets import DatasetClient

from src.config import Settings, get_settings
from src.eval.run_ragas import METRIC_KEYS, load_goldens, run_evaluation
from src.observability.tracing import init_langfuse
from src.rag.chain import RAGPipeline

log = logging.getLogger(__name__)

DATASET_NAME = "corporate_kb_goldens"
DATASET_DESCRIPTION = (
    "Эталонные вопросы/ответы корпоративной базы знаний "
    "(data/eval/goldens.json, записи без skip_rag)"
)
RUN_DESCRIPTION = "RAG-прогон по goldens с Ragas-scores (faithfulness, answer_relevancy, context_recall)"
EXPERIMENT_USER_ID = "experiment"

# Сколько ждать появления трейсов/scores в API после flush (секунды).
_INGEST_TIMEOUT_SEC = 60
_INGEST_POLL_SEC = 3


def ensure_dataset(lf: Langfuse) -> DatasetClient:
    """Возвращает датасет corporate_kb_goldens, создавая его при отсутствии."""
    try:
        dataset = lf.get_dataset(DATASET_NAME)
        print(f"Датасет '{DATASET_NAME}' уже существует (id={dataset.id}) — переиспользуем.")
        return dataset
    except Exception:
        log.info("Датасет %s не найден, создаём", DATASET_NAME)
    try:
        lf.create_dataset(name=DATASET_NAME, description=DATASET_DESCRIPTION)
        print(f"Датасет '{DATASET_NAME}' создан.")
        return lf.get_dataset(DATASET_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Не удалось создать/получить датасет '{DATASET_NAME}': {e}. "
            "Проверьте, что Langfuse доступен и ключи корректны."
        ) from e


def ensure_dataset_items(lf: Langfuse, goldens: List[Dict[str, Any]]) -> int:
    """
    Добавляет в датасет items из goldens без skip_rag.
    Идемпотентно: items, чей input.question уже есть в датасете, пропускаются.
    Возвращает число вновь созданных items.
    """
    eligible = [g for g in goldens if not g.get("skip_rag")]
    dataset = lf.get_dataset(DATASET_NAME)
    existing = {
        item.input.get("question")
        for item in dataset.items
        if isinstance(item.input, dict)
    }
    created = 0
    for g in eligible:
        if g["question"] in existing:
            continue
        try:
            lf.create_dataset_item(
                dataset_name=DATASET_NAME,
                input={"question": g["question"]},
                expected_output=g["answer"],
            )
            created += 1
        except Exception as e:
            print(f"  ! Ошибка создания item '{g['question'][:60]}': {e}")
    skipped = len(eligible) - created
    print(f"Items датасета: создано {created}, уже существовало {skipped} (всего подходящих {len(eligible)}).")
    return created


def run_experiment(
    lf: Langfuse, settings: Settings, dataset: DatasetClient, run_name: str
) -> int:
    """
    Прогоняет каждый item датасета через answer_traced с user_id/session_id
    эксперимента. Возвращает число успешно обработанных items.
    """
    pipeline = RAGPipeline(settings, lf=lf)
    ok = 0
    for idx, item in enumerate(dataset.items, 1):
        question = (item.input or {}).get("question", "")
        print(f"  [{idx}/{len(dataset.items)}] {question[:70]}")
        try:
            pipeline.answer_traced(
                question, user_id=EXPERIMENT_USER_ID, session_id=run_name
            )
            ok += 1
        except Exception as e:
            print(f"    ! Ошибка пайплайна на вопросе '{question[:60]}': {e}")
    try:
        lf.flush()
    except Exception as e:
        print(f"  ! Ошибка flush после прогона: {e}")
    return ok


def fetch_run_traces(lf: Langfuse, run_name: str, expected: int) -> Dict[str, str]:
    """
    Ждёт появления трейсов сессии в API и возвращает маппинг
    question -> trace_id по input root-спана ({"question": ...}).
    """
    deadline = time.time() + _INGEST_TIMEOUT_SEC
    question_to_trace: Dict[str, str] = {}
    while time.time() < deadline:
        try:
            traces = lf.api.trace.list(session_id=run_name, limit=100)
            question_to_trace = {
                t.input["question"]: t.id
                for t in traces.data
                if isinstance(t.input, dict) and t.input.get("question")
            }
        except Exception as e:
            print(f"  ! Ошибка чтения трейсов сессии: {e}")
        if len(question_to_trace) >= expected:
            break
        time.sleep(_INGEST_POLL_SEC)
    return question_to_trace


def link_traces_to_run(
    lf: Langfuse, dataset: DatasetClient, run_name: str, question_to_trace: Dict[str, str]
) -> int:
    """Привязывает трейсы к dataset run (обходной путь HW5 через api.dataset_run_items)."""
    linked = 0
    for item in dataset.items:
        question = (item.input or {}).get("question", "")
        trace_id = question_to_trace.get(question)
        if not trace_id:
            print(f"  ! Нет трейса для item '{question[:60]}' — пропуск привязки.")
            continue
        try:
            lf.api.dataset_run_items.create(
                run_name=run_name,
                run_description=RUN_DESCRIPTION,
                dataset_item_id=item.id,
                trace_id=trace_id,
            )
            linked += 1
        except Exception as e:
            print(f"  ! Ошибка привязки item '{question[:60]}': {e}")
    return linked


def push_ragas_scores(
    lf: Langfuse,
    results: Dict[str, Any],
    question_to_trace: Dict[str, str],
    run_name: str,
) -> int:
    """
    Пишет per-row метрики Ragas из run_evaluation(...).["details"] как scores
    на трейс соответствующего вопроса. None/NaN-значения пропускаются.
    Возвращает число созданных scores.
    """
    rows_by_question = {row["user_input"]: row for row in results.get("details", [])}
    created = 0
    for question, trace_id in question_to_trace.items():
        row = rows_by_question.get(question)
        if row is None:
            print(f"  ! Нет строки Ragas details для '{question[:60]}' — scores пропущены.")
            continue
        for metric in METRIC_KEYS:
            value = row.get(metric)
            if value is None:
                print(f"  ! {metric}=None для '{question[:60]}' — пропуск.")
                continue
            try:
                lf.create_score(
                    trace_id=trace_id,
                    name=metric,
                    value=float(value),
                    comment=f"Эксперимент: {run_name}",
                )
                created += 1
            except Exception as e:
                print(f"  ! Ошибка create_score {metric} для '{question[:60]}': {e}")
    try:
        lf.flush()
    except Exception as e:
        print(f"  ! Ошибка flush после scores: {e}")
    return created


def count_scored_traces(lf: Langfuse, trace_ids: List[str]) -> int:
    """Считает трейсы, у которых в API виден хотя бы один score (с ожиданием ingestion)."""
    deadline = time.time() + _INGEST_TIMEOUT_SEC
    remaining = set(trace_ids)
    while remaining and time.time() < deadline:
        for trace_id in list(remaining):
            try:
                scores = lf.api.scores.get_many(trace_id=trace_id, limit=50)
                if scores.data:
                    remaining.discard(trace_id)
            except Exception as e:
                print(f"  ! Ошибка чтения scores трейса {trace_id[:12]}…: {e}")
                remaining.discard(trace_id)
        if remaining:
            time.sleep(_INGEST_POLL_SEC)
    return len(trace_ids) - len(remaining)


def verify_run(
    lf: Langfuse, run_name: str, question_to_trace: Dict[str, str], expected_items: int
) -> bool:
    """Программная проверка результата через API. Печатает сводку, возвращает успех."""
    ok = True
    print("\n=== Проверка результата ===")
    try:
        dataset = lf.get_dataset(DATASET_NAME)
        n_items = len(dataset.items)
        status = "OK" if n_items == expected_items else "FAIL"
        ok = ok and n_items == expected_items
        print(f"  [{status}] Датасет '{DATASET_NAME}': items = {n_items} (ожидалось {expected_items})")
    except Exception as e:
        print(f"  [FAIL] Датасет '{DATASET_NAME}' недоступен: {e}")
        ok = False
    try:
        run = lf.api.datasets.get_run(DATASET_NAME, run_name)
        n_run_items = len(run.dataset_run_items or [])
        status = "OK" if n_run_items == expected_items else "FAIL"
        ok = ok and n_run_items == expected_items
        print(f"  [{status}] Dataset run '{run_name}': привязанных items = {n_run_items} (ожидалось {expected_items})")
    except Exception as e:
        print(f"  [FAIL] Dataset run '{run_name}' не найден: {e}")
        ok = False
    n_scored = count_scored_traces(lf, list(question_to_trace.values()))
    status = "OK" if n_scored == len(question_to_trace) else "FAIL"
    ok = ok and n_scored == len(question_to_trace)
    print(f"  [{status}] Трейсов со scores: {n_scored} из {len(question_to_trace)}")
    return ok


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    lf = init_langfuse(settings)
    if lf is None:
        print(
            "ОШИБКА: Langfuse-клиент не инициализирован. Проверьте, что сервер "
            "доступен (LANGFUSE_BASE_URL) и в .env заданы LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY."
        )
        return 1

    goldens = load_goldens(settings.goldens_path)
    expected_items = len([g for g in goldens if not g.get("skip_rag")])
    print(f"Goldens загружены: {len(goldens)} записей, в эксперимент пойдёт {expected_items} (без skip_rag).")

    # 1. Датасет + items (идемпотентно).
    ensure_dataset(lf)
    ensure_dataset_items(lf, goldens)
    dataset = lf.get_dataset(DATASET_NAME)

    # 2. Прогон эксперимента и привязка трейсов к dataset run.
    run_name = f"rag_run_{datetime.now():%Y%m%d_%H%M%S}"
    print(f"\nПрогон эксперимента '{run_name}' по {len(dataset.items)} items…")
    n_answered = run_experiment(lf, settings, dataset, run_name)
    print(f"Прогон завершён: {n_answered}/{len(dataset.items)} ответов получено.")

    question_to_trace = fetch_run_traces(lf, run_name, expected=n_answered)
    print(f"Трейсов сессии найдено в API: {len(question_to_trace)}")
    linked = link_traces_to_run(lf, dataset, run_name, question_to_trace)
    print(f"Привязано к dataset run: {linked}/{len(dataset.items)} items.")

    # 3. Ragas-оценка (один прогон) и scores на трейсы.
    print("\nЗапуск Ragas-оценки (несколько минут)…")
    try:
        results = run_evaluation(settings)
    except Exception as e:
        print(f"ОШИБКА: Ragas-оценка упала: {e}. Трейсы и dataset run уже созданы; scores не выставлены.")
        return 1
    print(
        "Ragas средние (без OOD): "
        + ", ".join(f"{k}={results.get(k)}" for k in METRIC_KEYS)
    )
    n_scores = push_ragas_scores(lf, results, question_to_trace, run_name)
    print(f"Создано scores: {n_scores}")

    # 4. Программная проверка.
    success = verify_run(lf, run_name, question_to_trace, expected_items)
    base = settings.langfuse_base_url.rstrip("/")
    print(f"\nUI: {base}  (Datasets → {DATASET_NAME} → Runs → {run_name}; Traces — сессия {run_name})")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
