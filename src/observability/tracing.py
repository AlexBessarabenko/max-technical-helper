"""Langfuse-трейсинг RAG-пайплайна: клиент + безопасные обёртки над SDK.

Обёртки гарантируют, что сбой Langfuse (недоступный сервер, ошибка
сериализации) никогда не роняет ответ пользователю, а ошибки бизнес-кода
(retrieve, вызов LLM) пробрасываются вызывающему без изменений.
"""

import logging
import sys
from contextlib import contextmanager

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


@contextmanager
def langfuse_scope(factory, what: str):
    """
    Безопасно входит в контекстный менеджер Langfuse (observation,
    propagate_attributes): `factory` создаёт CM, вход выполняется здесь.
    Ошибки создания/входа/выхода CM подавляются — блок получает None и
    выполняется без этой observation. Исключения ТЕЛА блока (retrieve,
    вызов LLM) не подавляются: __exit__ получает их exc_info (span
    помечается ошибкой), после чего они пробрасываются вызывающему.
    """
    cm = None
    try:
        cm = factory()
        obs = cm.__enter__()
    except Exception:
        log.warning("Langfuse: %s — не открыто, работаем без трейса", what, exc_info=True)
        cm = obs = None
    try:
        yield obs
    except BaseException:
        if cm is not None:
            try:
                cm.__exit__(*sys.exc_info())
            except Exception:
                log.warning("Langfuse: %s — ошибка при закрытии", what, exc_info=True)
        raise
    if cm is not None:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            log.warning("Langfuse: %s — ошибка при закрытии", what, exc_info=True)


@contextmanager
def suppress_langfuse_errors(what: str):
    """Подавляет ошибки одного вызова Langfuse SDK (update / event / flush)."""
    try:
        yield
    except Exception:
        log.warning("Langfuse: %s — ошибка, продолжаем без трейса", what, exc_info=True)
