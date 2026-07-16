"""Инициализация Langfuse-клиента для трейсинга RAG-пайплайна."""

import logging

from langfuse import Langfuse

log = logging.getLogger(__name__)


def init_langfuse(settings) -> "Langfuse | None":
    """
    Создаёт Langfuse-клиент по ключам из настроек.
    Возвращает None, если ключи не заданы или сервер недоступен —
    в этом случае пайплайн работает без трейсинга.
    """
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        lf = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
        )
        return lf if lf.auth_check() else None
    except Exception:
        log.warning("Langfuse недоступен, трейсинг отключён", exc_info=True)
        return None
