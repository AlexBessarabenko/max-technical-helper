"""RAG-цепочка: retrieval из ChromaDB + генерация ответа через YandexGPT."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chromadb
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langfuse import propagate_attributes

from src.config import Settings
from src.observability.tracing import langfuse_scope, suppress_langfuse_errors
from src.rag.embeddings import YandexEmbeddings

log = logging.getLogger(__name__)

NO_ANSWER_TEXT = (
    "К сожалению, в базе знаний нет информации по этому вопросу. "
    "Рекомендую обратиться в службу поддержки: helpdesk@technosphere.example, вн. 1001."
)

SYSTEM_PROMPT = """Ты — внутренний ассистент компании «ТехноСфера» (IT и HR поддержка).
Отвечай ТОЛЬКО на основе контекста ниже, ничего не выдумывай.
Если в контексте есть точный ответ — ответь по контексту.
Если точного ответа нет — ответь строго по шаблону:
фраза «К сожалению, в базе знаний нет информации по этому вопросу.» и сразу за ней
список из 4–6 конкретных фактов из контекста по ближайшей теме (цифры, сроки, шаги).
Не добавляй ничего от себя: ни рекомендаций, ни контактов, ни предложений помочь.
Стиль: вежливый корпоративный, начинай со "Здравствуйте!", по делу, со структурой.

Пример ответа при частичном совпадении темы:
«Здравствуйте! К сожалению, в базе знаний нет информации по этому вопросу.
По ближайшей теме (обычные командировки) в базе указано:
- заявка создаётся в 1С:ЗУП не позднее чем за 5 рабочих дней до выезда;
- суточные — 1500 руб./день по России и 2000 руб./день для Москвы и Санкт-Петербурга;
- аванс перечисляется на зарплатную карту за 3 рабочих дня до выезда;
- авансовый отчёт сдаётся в течение 5 рабочих дней после возвращения;
- лимит проживания — до 6000 руб./ночь по России.»

Контекст:
{context}"""

# Сколько последних пар реплик (user/assistant) подставлять в запрос.
_HISTORY_PAIRS = 3


class YandexChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI с параметрами, совместимыми с YandexGPT OpenAI-совместимым API.
    YandexGPT не принимает ряд полей (n, stop, stream, logprobs и др.),
    которые langchain-openai добавляет в запрос по умолчанию.
    """

    @property
    def _default_params(self) -> Dict[str, object]:
        params = super()._default_params
        # YandexGPT не поддерживает ряд параметров OpenAI API.
        unsupported = (
            "n", "stop", "stream", "logprobs", "top_logprobs", "logit_bias",
            "extra_body", "reasoning_effort", "reasoning", "verbosity",
            "context_management", "include", "prompt_cache_options",
            "service_tier", "truncation", "store",
        )
        for key in unsupported:
            params.pop(key, None)
        # Очень маленькие значения температуры YandexGPT может отклонять
        # из-за научной нотации.
        temperature = params.get("temperature")
        if temperature is not None and temperature < 1e-6:
            params["temperature"] = 0.0
        return params


@dataclass
class RAGResult:
    answer: str
    sources: List[str]
    contexts: List[str]
    status: str  # "success" | "no_answer"


class RAGPipeline:
    # Заглушка для юнит-тестов, создающих пайплайн через __new__ без __init__.
    _embeddings: Optional[YandexEmbeddings] = None

    def __init__(self, settings: Settings, lf=None):
        self.settings = settings
        self._lf = lf  # Langfuse-клиент для трейсинга (None — трейсинг выключен)
        self._model_name = f"gpt://{settings.yandex_folder_id}/{settings.yandex_llm_model}"
        self._embeddings = YandexEmbeddings(
            api_key=settings.yandex_api_key,
            folder_id=settings.yandex_folder_id,
        )
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._llm = YandexChatOpenAI(
            model=self._model_name,
            api_key=settings.yandex_api_key,
            base_url=settings.yandex_base_url,
            temperature=0.1,
            timeout=30,
            max_retries=3,
        )

    def retrieve(
        self, question: str, source_filter: Optional[str] = None
    ) -> List[Tuple[str, dict, float]]:
        """
        Ищет релевантные чанки в ChromaDB.
        Возвращает список (текст, метаданные, score), отфильтрованный
        по score = 1 - distance >= retrieval_min_score.
        """
        if self._embeddings is not None:
            query_embedding = self._embeddings.embed_query(question)
        else:
            # Юнит-тесты с замоканной коллекцией: содержимое эмбеддинга неважно.
            query_embedding = [0.0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=self.settings.retrieval_top_k,
            where={"source": source_filter} if source_filter else None,
        )
        hits = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1 - distance
            if score >= self.settings.retrieval_min_score:
                hits.append((text, metadata, score))
        return hits

    def _no_answer_result(self) -> RAGResult:
        return RAGResult(answer=NO_ANSWER_TEXT, sources=[], contexts=[], status="no_answer")

    def _build_prompt(
        self,
        question: str,
        hits: List[Tuple[str, dict, float]],
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> List:
        """Собирает сообщения запроса: system с контекстом, история, вопрос."""
        context = "\n\n".join(
            f"{i}. [Источник: {metadata.get('title', '')}]\n{text}"
            for i, (text, metadata, _) in enumerate(hits, 1)
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
        for role, text in (chat_history or [])[-_HISTORY_PAIRS * 2:]:
            if role == "user":
                messages.append(HumanMessage(content=text))
            else:
                messages.append(AIMessage(content=text))
        messages.append(HumanMessage(content=question))
        return messages

    def _build_result(self, answer_text: str, hits: List[Tuple[str, dict, float]]) -> RAGResult:
        sources = list(dict.fromkeys(
            metadata.get("title", "") for _, metadata, _ in hits if metadata.get("title")
        ))
        return RAGResult(
            answer=answer_text,
            sources=sources,
            contexts=[text for text, _, _ in hits],
            status="success",
        )

    def answer(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> RAGResult:
        """
        Отвечает на вопрос по базе знаний.
        chat_history — список пар (role, text), role ∈ {"user", "assistant"};
        в запрос подставляются последние 3 пары реплик.
        """
        hits = self.retrieve(question)
        if not hits:
            return self._no_answer_result()

        messages = self._build_prompt(question, hits, chat_history)
        response = self._llm.invoke(messages)
        return self._build_result(response.content, hits)

    def answer_traced(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        user_id: str = "unknown",
        session_id: str = "unknown",
    ) -> RAGResult:
        """
        Обёртка над answer() с Langfuse-трейсом: span'ы preprocessing →
        generation → postprocessing, событие pipeline_completed, user_id и
        session_id на трейсе. Вызов LLM — строго внутри generation-observation,
        чтобы длительность generation отражала чистое время LLM.
        Ошибки Langfuse SDK (открытие/апдейт observation, event, flush)
        подавляются — ответ возвращается в любом случае. Ошибки retrieve/LLM
        НЕ перехватываются и пробрасываются вызывающему, как в answer().
        При lf=None — обычный answer().
        """
        lf = getattr(self, "_lf", None)
        if lf is None:
            return self.answer(question, chat_history)
        with langfuse_scope(
            lambda: propagate_attributes(user_id=user_id, session_id=session_id),
            "propagate_attributes",
        ):
            with langfuse_scope(
                lambda: lf.start_as_current_observation(
                    name="max_request", as_type="span", input={"question": question}
                ),
                "max_request",
            ) as root:
                with langfuse_scope(
                    lambda: lf.start_as_current_observation(name="preprocessing", as_type="span"),
                    "preprocessing",
                ) as span:
                    hits = self.retrieve(question)
                    if span is not None:
                        with suppress_langfuse_errors("preprocessing.update"):
                            span.update(output={"num_contexts": len(hits)})
                if not hits:
                    result = self._no_answer_result()
                else:
                    prompt = self._build_prompt(question, hits, chat_history)
                    # generation-observation открывается НЕПОСРЕДСТВЕННО вокруг вызова
                    # LLM; сам вызов намеренно не защищён — его ошибки идут вызывающему.
                    with langfuse_scope(
                        lambda: lf.start_as_current_observation(
                            name="llm_generation",
                            as_type="generation",
                            model=getattr(self, "_model_name", None),
                            input=prompt,
                        ),
                        "llm_generation",
                    ) as gen:
                        resp = self._llm.invoke(prompt)
                        token_usage = (resp.response_metadata or {}).get("token_usage") or {}
                        usage_details = {
                            k: v
                            for k, v in {
                                "input": token_usage.get("prompt_tokens"),
                                "output": token_usage.get("completion_tokens"),
                            }.items()
                            if v is not None
                        }
                        if gen is not None:
                            with suppress_langfuse_errors("llm_generation.update"):
                                gen.update(output=resp.content, usage_details=usage_details)
                    with langfuse_scope(
                        lambda: lf.start_as_current_observation(name="postprocessing", as_type="span"),
                        "postprocessing",
                    ) as span:
                        result = self._build_result(resp.content, hits)
                        if span is not None:
                            with suppress_langfuse_errors("postprocessing.update"):
                                span.update(output={"status": result.status})
                if root is not None:
                    with suppress_langfuse_errors("pipeline_completed event"):
                        lf.create_event(name="pipeline_completed", metadata={"status": result.status})
                    with suppress_langfuse_errors("max_request.update"):
                        root.update(output={"answer": result.answer[:500], "status": result.status})
            with suppress_langfuse_errors("flush"):
                lf.flush()
        return result
