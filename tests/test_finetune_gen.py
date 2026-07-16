from src.data_gen.finetune_gen import build_dataset


def test_dataset_format_and_size():
    ds = build_dataset()
    assert 150 <= len(ds) <= 300
    for row in ds:
        msgs = row["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert all(m["content"].strip() for m in msgs)


def test_dataset_no_duplicates_and_corporate_style():
    ds = build_dataset()
    pairs = [(m["messages"][1]["content"], m["messages"][2]["content"]) for m in ds]
    assert len(set(pairs)) == len(pairs), "дубликаты пар"
    greetings = sum(1 for _, a in pairs if a.startswith("Здравствуйте"))
    assert greetings / len(pairs) >= 0.8, "ответы должны начинаться с корпоративного приветствия"


def test_dataset_deterministic():
    assert build_dataset() == build_dataset(), "генерация должна быть детерминированной"
