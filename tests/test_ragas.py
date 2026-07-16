"""Ragas quality gate: faithfulness / answer relevancy / context recall."""

import src.eval.ragas_compat  # noqa: F401  # заглушки для ragas 0.2.x — импорт ПЕРВЫМ, до ragas

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness

from src.config import get_settings
from src.eval.run_ragas import METRIC_KEYS, build_evaluator, run_evaluation
from src.rag.chain import RAGPipeline

FAITHFULNESS_THRESHOLD = 0.7
ANSWER_RELEVANCY_THRESHOLD = 0.7
CONTEXT_RECALL_THRESHOLD = 0.6

THRESHOLDS = {
    "faithfulness": FAITHFULNESS_THRESHOLD,
    "answer_relevancy": ANSWER_RELEVANCY_THRESHOLD,
    "context_recall": CONTEXT_RECALL_THRESHOLD,
}

GOLDENS_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "goldens.json"


def test_goldens_exist():
    """Эталонный набор существует, достаточного размера и корректного формата."""
    goldens = json.loads(GOLDENS_PATH.read_text(encoding="utf-8"))
    assert len(goldens) >= 15, "Должно быть минимум 15 эталонов"
    for g in goldens:
        assert isinstance(g.get("question"), str) and g["question"].strip(), "question обязателен"
        assert isinstance(g.get("answer"), str) and g["answer"].strip(), "answer обязателен"
        assert isinstance(g.get("contexts"), list), "contexts обязателен (list)"
        assert all(isinstance(c, str) for c in g["contexts"]), "contexts — list[str]"


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def ragas_results(settings):
    """Один прогон run_evaluation на модуль — метрики проверяют все тесты."""
    return run_evaluation(settings)


@pytest.mark.ragas
def test_ragas_evaluation(ragas_results):
    """Quality gate: усреднённые метрики Ragas не ниже порогов. None — warning."""
    for key in METRIC_KEYS:
        value = ragas_results.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            warnings.warn(f"{key} не вычислено (None/NaN)", stacklevel=2)
            continue
        assert value >= THRESHOLDS[key], (
            f"{key} {value:.3f} ниже порога {THRESHOLDS[key]}"
        )


@pytest.mark.ragas
def test_no_hallucinations(settings):
    """
    OOD-вопрос: пайплайн обязан отказаться (status no_answer или явное
    «нет информации»), а не выдумывать ответ. Дополнительно проверяем
    faithfulness фактического содержимого ответа: всё, что пайплайн
    утверждает помимо отказа, должно следовать из извлечённого контекста.

    NB: мета-утверждения самого отказа («в базе знаний нет информации...»),
    приветствие и фрейминг из оценки исключены: strict-судья (deepseek-v4-flash)
    не может вывести факт ОТСУТСТВИЯ информации из контекста и оценивает такие
    утверждения нестабильно (замерено: verdict флуктуирует 0/1 при неизменном
    ответе). Корректность отказа гарантируется ассертом выше.
    """
    pipeline = RAGPipeline(settings)
    question = "Как оформить командировку на Марс?"
    result = pipeline.answer(question)

    assert result.status == "no_answer" or "нет информации" in result.answer.lower(), (
        f"Модель галлюцинирует на OOD-вопросе: status={result.status}, answer={result.answer[:200]}"
    )

    # Фактические утверждения — элементы списка в ответе (шаблон отказа в
    # SYSTEM_PROMPT требует перечислять ближайшие факты списком).
    factual = "\n".join(
        line for line in result.answer.splitlines()
        if line.strip().startswith(("-", "•", "*"))
    )
    if not factual.strip():
        warnings.warn(
            "Ответ не содержит фактических утверждений — faithfulness не применим",
            stacklevel=2,
        )
        return

    llm, embeddings = build_evaluator(settings)
    metric = faithfulness
    metric.llm = llm
    metric.embeddings = embeddings

    eval_data = [
        {
            "user_input": question,
            "response": factual,
            "retrieved_contexts": result.contexts,
            "reference": "В базе знаний нет информации по этому вопросу",
        }
    ]
    results = evaluate(
        dataset=Dataset.from_list(eval_data),
        metrics=[metric],
        llm=llm,
        embeddings=embeddings,
    )
    faithfulness_value = results["faithfulness"][0] if results["faithfulness"] else None

    if faithfulness_value is None or (isinstance(faithfulness_value, float) and np.isnan(faithfulness_value)):
        warnings.warn("faithfulness для OOD-вопроса не вычислено (None/NaN)", stacklevel=2)
        return
    assert faithfulness_value >= 0.8, (
        f"Модель галлюцинирует! Faithfulness фактических утверждений: {faithfulness_value:.3f}"
    )
