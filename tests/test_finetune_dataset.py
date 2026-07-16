"""Fast structural checks for data/finetune/dataset.jsonl (no model download)."""

import json
from pathlib import Path

import pytest

from src.config import get_settings

DATASET_PATH = get_settings().finetune_dataset_path


def _records() -> list[dict]:
    with Path(DATASET_PATH).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_dataset_structure() -> None:
    records = _records()
    assert len(records) > 0, "dataset.jsonl is empty"
    for i, record in enumerate(records):
        messages = record.get("messages")
        assert isinstance(messages, list), f"line {i}: no messages list"
        assert [m["role"] for m in messages] == [
            "system",
            "user",
            "assistant",
        ], f"line {i}: unexpected roles {[m.get('role') for m in messages]}"
        for m in messages:
            assert set(m) == {"role", "content"}, f"line {i}: extra keys {set(m)}"
            assert isinstance(m["content"], str) and m["content"].strip(), (
                f"line {i}: empty content"
            )


def test_dataset_loads_with_hf_datasets() -> None:
    datasets = pytest.importorskip("datasets")
    ds = datasets.load_dataset("json", data_files=str(DATASET_PATH), split="train")
    assert len(ds) == len(_records())
    first = ds[0]["messages"]
    assert [m["role"] for m in first] == ["system", "user", "assistant"]
