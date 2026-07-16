from src.rag.indexing import chunk_markdown, parse_frontmatter


def test_parse_frontmatter():
    meta, body = parse_frontmatter("---\nsource: IT\ntitle: VPN\n---\n# VPN\nТекст")
    assert meta == {"source": "IT", "title": "VPN"}
    assert body.startswith("# VPN")


def test_chunk_markdown_splits_by_headers_and_size():
    body = "# Статья\n" + "\n".join(f"## Раздел {i}\n" + "текст " * 100 for i in range(6))
    chunks = chunk_markdown(body, max_chars=1200, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1300 for c in chunks)
    assert chunks[0].startswith("# Статья")


def test_chunk_markdown_overlap_present():
    body = "## A\n" + "а" * 900 + "\n## B\n" + "б" * 900
    chunks = chunk_markdown(body, max_chars=1000, overlap=100)
    assert len(chunks) == 2
