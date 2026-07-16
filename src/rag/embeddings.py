"""Обёртка над родным Yandex Text Embedding API."""

import threading
import time
from typing import List

import requests
from langchain_core.embeddings import Embeddings


class YandexEmbeddings(Embeddings):
    """
    Обёртка над родным Yandex Text Embedding API.
    Использует разные модели для документов (text-search-doc) и запросов
    (text-search-query), как рекомендуется в документации Yandex.
    """

    def __init__(self, api_key: str, folder_id: str, timeout: float = 60.0):
        self.api_key = api_key
        self.folder_id = folder_id
        self.timeout = timeout
        self.url = "https://ai.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "x-folder-id": folder_id,
        }
        # Yandex Embeddings API возвращает 429 при слишком частых запросах.
        # Делаем запросы последовательными с небольшой паузой.
        self._lock = threading.Lock()
        self._delay = 0.3
        self._max_retries = 3

    def _model_uri(self, text_type: str) -> str:
        model = "text-search-doc" if text_type == "doc" else "text-search-query"
        return f"emb://{self.folder_id}/{model}/latest"

    def _embed(self, text: str, text_type: str) -> List[float]:
        with self._lock:
            response = self._request_with_retry(text, text_type)
            time.sleep(self._delay)
            return response.json()["embedding"]

    def _request_with_retry(self, text: str, text_type: str) -> requests.Response:
        """POST с retry: до 3 повторов с backoff 2/4/8 с на 429 и 5xx."""
        backoff = 2
        for attempt in range(self._max_retries + 1):
            response = requests.post(
                self.url,
                json={"modelUri": self._model_uri(text_type), "text": text},
                headers=self.headers,
                timeout=self.timeout,
            )
            if response.ok:
                return response
            retriable = response.status_code == 429 or response.status_code >= 500
            if not retriable or attempt == self._max_retries:
                response.raise_for_status()
            time.sleep(backoff)
            backoff *= 2
        raise RuntimeError("unreachable")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t, "doc") for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text, "query")
