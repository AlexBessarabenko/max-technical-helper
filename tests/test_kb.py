from pathlib import Path
import re

KB_DIR = Path("data/kb")


def _parse(path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    assert m, f"{path.name}: нет YAML-frontmatter"
    return m.group(1), m.group(2)


def test_kb_count_and_balance():
    files = list(KB_DIR.glob("*.md"))
    assert len(files) >= 25, "минимум 25 статей"
    it = sum(1 for f in files if _parse(f)[0].strip().startswith("source: IT"))
    hr = sum(1 for f in files if _parse(f)[0].strip().startswith("source: HR"))
    assert it >= 10 and hr >= 10


def test_kb_frontmatter_and_length():
    for f in KB_DIR.glob("*.md"):
        fm, body = _parse(f)
        assert re.search(r"^source: (IT|HR)$", fm, re.M), f"{f.name}: source должен быть IT или HR"
        assert re.search(r"^title: .+", fm, re.M), f"{f.name}: нет title"
        assert len(body) >= 800, f"{f.name}: статья короче 800 символов"
        assert "##" in body, f"{f.name}: нужны подзаголовки ##"
