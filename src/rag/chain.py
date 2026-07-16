"""RAG-цепочка: retrieval из ChromaDB + генерация ответа через YandexGPT."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chromadb
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import Settings
from src.rag.embeddings import YandexEmbeddings

NO_ANSWER_TEXT = (
    "К сожалению, в базе знаний нет информации по этому вопросу. "
    "Рекомендую обратиться в службу поддержки: helpdesk@technosphere.example, вн. 1001."
)

SYSTEM_PROMPT = """Ты — внутренний ассистент компании «ТехноСфера» (IT и HR поддержка).
Отвечай ТОЛЬКО на основе контекста ниже. Если в контексте нет ответа —
честно скажи, что информации нет, и предложи обратиться в поддержку.
Стиль: вежливый корпоративный, начинай со "Здравствуйте!", по делу, со структурой.

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
        self._lf = lf  # Langfuse-клиент, понадобится в задаче трейсинга
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
            model=f"gpt://{settings.yandex_folder_id}/{settings.yandex_llm_model}",
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
            return RAGResult(answer=NO_ANSWER_TEXT, sources=[], contexts=[], status="no_answer")

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

        response = self._llm.invoke(messages)
        sources = list(dict.fromkeys(
            metadata.get("title", "") for _, metadata, _ in hits if metadata.get("title")
        ))
        return RAGResult(
            answer=response.content,
            sources=sources,
            contexts=[text for text, _, _ in hits],
            status="success",
        )
