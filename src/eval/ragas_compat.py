"""
Совместимость ragas 0.2.x с langchain-community >= 0.4 и CPython 3.14.

Ragas 0.2.x импортирует ChatVertexAI/VertexAI из старых путей
langchain_community.chat_models.vertexai и langchain_community.llms.vertexai,
которые были удалены в новых версиях langchain-community.

Также ragas вызывает ``nest_asyncio.apply()`` при импорте исполнителя.
В CPython 3.14 ``asyncio.wait_for`` реализован через ``asyncio.timeout``,
а ``nest_asyncio`` не поддерживает ``asyncio.current_task()`` внутри
корутины, запущенной через ``asyncio.run`` (возвращает ``None``). Это
приводит к ``RuntimeError: Timeout should be used inside a task`` и NaN
для всех метрик. Отключая ``nest_asyncio.apply()``, мы оставляем стандартный
``asyncio.run``, который корректно создаёт Task и current_task.
"""
import sys
import types


def _install_stubs() -> None:
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        mod = types.ModuleType("langchain_community.chat_models.vertexai")
        mod.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[misc]
        sys.modules["langchain_community.chat_models.vertexai"] = mod

    if "langchain_community.llms.vertexai" not in sys.modules:
        mod = types.ModuleType("langchain_community.llms.vertexai")
        mod.VertexAI = type("VertexAI", (), {})  # type: ignore[misc]
        sys.modules["langchain_community.llms.vertexai"] = mod


def _disable_nest_asyncio() -> None:
    """
    Делаем ``nest_asyncio.apply()`` no-op до импорта ragas.

    ragas импортирует nest_asyncio и вызывает ``nest_asyncio.apply()`` на
    уровне модуля ``ragas.executor``. В CPython 3.14 patched event loop
    теряет ``current_task()``, что ломает ``asyncio.wait_for``.
    """
    try:
        import nest_asyncio
    except ImportError:
        return

    def _noop_apply(loop=None) -> None:  # noqa: ARG001
        return None

    nest_asyncio.apply = _noop_apply


_install_stubs()
_disable_nest_asyncio()
