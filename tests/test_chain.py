from unittest.mock import MagicMock

from src.rag.chain import RAGPipeline


def _pipeline(chroma_results, llm_text="Ответ по контексту"):
    p = RAGPipeline.__new__(RAGPipeline)
    p.settings = MagicMock(retrieval_top_k=4, retrieval_min_score=0.25)
    p._collection = MagicMock()
    p._collection.query.return_value = chroma_results
    p._llm = MagicMock()
    p._llm.invoke.return_value = MagicMock(content=llm_text)
    return p


def test_retrieve_filters_by_score():
    p = _pipeline({
        "documents": [["хороший чанк", "плохой чанк"]],
        "metadatas": [[{"title": "VPN", "source": "IT", "doc_id": "it_vpn_setup"}, {"title": "X", "source": "IT", "doc_id": "x"}]],
        "distances": [[0.3, 0.9]],   # distance = 1 - score для cosine
    })
    hits = p.retrieve("как настроить vpn")
    assert [h[0] for h in hits] == ["хороший чанк"]  # score 0.7 >= 0.25; 0.1 отфильтрован


def test_answer_no_context_returns_refusal_without_llm():
    p = _pipeline({"documents": [[]], "metadatas": [[]], "distances": [[]]})
    r = p.answer("что-то совсем не по теме")
    assert r.status == "no_answer"
    assert r.contexts == []
    assert "нет информации" in r.answer.lower()
    p._llm.invoke.assert_not_called()


def test_answer_success_includes_sources():
    p = _pipeline({
        "documents": [["VPN настраивается через OpenVPN..."]],
        "metadatas": [[{"title": "Настройка VPN", "source": "IT", "doc_id": "it_vpn_setup"}]],
        "distances": [[0.2]],
    })
    r = p.answer("как настроить vpn")
    assert r.status == "success"
    assert "Настройка VPN" in r.sources
    assert "VPN настраивается через OpenVPN..." in r.contexts
