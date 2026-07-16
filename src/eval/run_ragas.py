"""Прогон Ragas-оценки качества RAG-пайплайна по эталонному набору goldens."""

import src.eval.ragas_compat  # noqa: F401  # заглушки для ragas 0.2.x — импорт ПЕРВЫМ, до ragas

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_recall, faithfulness

from src.config import Settings
from src.rag.chain import RAGPipeline, YandexChatOpenAI
from src.rag.embeddings import YandexEmbeddings

log = logging.getLogger(__name__)

# Ключи метрик в результате (совпадают с именами метрик ragas).
METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_recall")

RESULTS_PATH = Path(__file__).resolve().parents[2] / "tests" / "ragas_results.json"


def load_goldens(path: Path) -> List[Dict[str, Any]]:
    """Читает goldens.json."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class YandexSymmetricEmbeddings(YandexEmbeddings):
    """
    Симметричные эмбеддинги для Ragas-метрик сравнения текстов одного типа.

    Ragas (answer_relevancy) сравнивает embed_query(вопрос) с
    embed_documents(сгенерированные вопросы), предполагая симметричную модель.
    Родные Yandex-модели асимметричны (text-search-query vs text-search-doc),
    поэтому косинусная близость даже близких парафраз занижается (~0.47 вместо
    ~0.83). Для сравнения «вопрос–вопрос» используем одну (doc) модель.
    Применяется только в evaluator; retrieval пайплайна не затрагивается.
    """

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text, "doc")


def build_evaluator(settings: Settings) -> Tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Evaluator LLM (YandexChatOpenAI) и embeddings для Ragas."""
    llm = YandexChatOpenAI(
        model=f"gpt://{settings.yandex_folder_id}/{settings.yandex_llm_model}",
        api_key=settings.yandex_api_key,
        base_url=settings.yandex_base_url,
        timeout=120,
        max_retries=3,
    )
    embeddings = YandexSymmetricEmbeddings(
        api_key=settings.yandex_api_key,
        folder_id=settings.yandex_folder_id,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def _mean(values: List[Any]) -> Optional[float]:
    """Среднее с игнорированием None/NaN; None, если значений нет."""
    clean = [float(v) for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(clean)) if clean else None


def _score_or_none(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def run_evaluation(settings: Settings) -> Dict[str, Any]:
    """
    Прогоняет goldens без skip_rag через RAGPipeline.answer и считает метрики
    Ragas (faithfulness, answer_relevancy, context_recall) с evaluator на
    YandexChatOpenAI/YandexEmbeddings. Усредняет с игнорированием None/NaN,
    пишет tests/ragas_results.json и возвращает dict с теми же ключами + details.
    """
    goldens = [g for g in load_goldens(settings.goldens_path) if not g.get("skip_rag")]
    pipeline = RAGPipeline(settings)
    llm, embeddings = build_evaluator(settings)

    metrics = [faithfulness, answer_relevancy, context_recall]
    for metric in metrics:
        metric.llm = llm
        metric.embeddings = embeddings

    rows = []
    statuses = []
    for golden in goldens:
        result = pipeline.answer(golden["question"])
        rows.append(
            {
                "user_input": golden["question"],
                "response": result.answer,
                "retrieved_contexts": result.contexts,
                "reference": golden["answer"],
            }
        )
        statuses.append(result.status)

    results = evaluate(
        dataset=Dataset.from_list(rows),
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    averaged = {key: _mean(list(results[key])) for key in METRIC_KEYS}
    details = []
    for i, row in enumerate(rows):
        scores = {key: _score_or_none(results[key][i]) for key in METRIC_KEYS}
        details.append({**row, "status": statuses[i], **scores})

    results_dict: Dict[str, Any] = {**averaged, "details": details}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    log.info("Ragas результаты записаны в %s", RESULTS_PATH)

    return results_dict
