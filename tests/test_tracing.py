from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.observability.tracing import init_langfuse
from src.rag.chain import RAGPipeline


def _pipeline(chroma_results, llm_text="Ответ по контексту", lf=None):
    p = RAGPipeline.__new__(RAGPipeline)
    p.settings = MagicMock(retrieval_top_k=4, retrieval_min_score=0.25)
    p._collection = MagicMock()
    p._collection.query.return_value = chroma_results
    p._llm = MagicMock()
    p._llm.invoke.return_value = MagicMock(content=llm_text)
    p._lf = lf
    p._model_name = "test-model"
    return p


_HITS = {
    "documents": [["VPN настраивается через OpenVPN..."]],
    "metadatas": [[{"title": "Настройка VPN", "source": "IT", "doc_id": "it_vpn_setup"}]],
    "distances": [[0.2]],
}


def test_init_langfuse_without_keys_returns_none():
    s = Settings(langfuse_public_key="", langfuse_secret_key="")
    assert init_langfuse(s) is None


def test_init_langfuse_returns_none_when_auth_fails():
    s = Settings(langfuse_public_key="pk", langfuse_secret_key="sk")
    with patch("src.observability.tracing.Langfuse") as cls:
        cls.return_value.auth_check.return_value = False
        assert init_langfuse(s) is None


def test_init_langfuse_returns_none_on_constructor_error():
    s = Settings(langfuse_public_key="pk", langfuse_secret_key="sk")
    with patch("src.observability.tracing.Langfuse", side_effect=RuntimeError("boom")):
        assert init_langfuse(s) is None


def test_answer_traced_without_lf_behaves_like_answer():
    p = _pipeline(_HITS, lf=None)
    r = p.answer_traced("как настроить vpn", user_id="u1", session_id="s1")
    assert r.status == "success"
    assert "Настройка VPN" in r.sources
    p._llm.invoke.assert_called_once()


def test_answer_traced_with_lf_returns_result_and_flushes():
    lf = MagicMock()
    p = _pipeline(_HITS, lf=lf)
    r = p.answer_traced("как настроить vpn", user_id="u1", session_id="s1")
    assert r.status == "success"
    p._llm.invoke.assert_called_once()
    lf.flush.assert_called_once()
    lf.create_event.assert_called_once()


def test_answer_traced_no_answer_without_llm():
    lf = MagicMock()
    p = _pipeline({"documents": [[]], "metadatas": [[]], "distances": [[]]}, lf=lf)
    r = p.answer_traced("что-то не по теме", user_id="u1", session_id="s1")
    assert r.status == "no_answer"
    p._llm.invoke.assert_not_called()
    lf.flush.assert_called_once()


def test_answer_traced_lf_error_still_returns_answer():
    # Langfuse бросает из всех методов → ответ всё равно success,
    # LLM вызван ровно один раз (никакого дублирующего прогона).
    lf = MagicMock()
    lf.start_as_current_observation.side_effect = RuntimeError("langfuse down")
    lf.create_event.side_effect = RuntimeError("langfuse down")
    lf.flush.side_effect = RuntimeError("langfuse down")
    p = _pipeline(_HITS, lf=lf)
    r = p.answer_traced("как настроить vpn", user_id="u1", session_id="s1")
    assert r.status == "success"
    assert "Настройка VPN" in r.sources
    p._llm.invoke.assert_called_once()


def test_answer_traced_llm_error_propagates():
    # Ошибка LLM не подавляется и не маскируется под ошибку трейсинга.
    lf = MagicMock()
    p = _pipeline(_HITS, lf=lf)
    p._llm.invoke.side_effect = RuntimeError("LLM down")
    with pytest.raises(RuntimeError, match="LLM down"):
        p.answer_traced("как настроить vpn", user_id="u1", session_id="s1")
    lf.flush.assert_not_called()
