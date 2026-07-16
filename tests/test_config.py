from src.config import get_settings


def test_settings_defaults():
    s = get_settings()
    assert s.yandex_llm_model == "yandexgpt"
    assert s.retrieval_top_k == 4
    assert str(s.kb_dir).endswith("data/kb")
