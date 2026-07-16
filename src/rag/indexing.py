"""Индексация базы знаний: frontmatter, heading-aware чанкинг, запись в ChromaDB."""

import re
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterator

import chromadb
import yaml

from src.config import Settings
from src.rag.embeddings import YandexEmbeddings

DEFAULT_MAX_CHARS = 800
DEFAULT_OVERLAP = 100
_BATCH_SIZE = 16
_HEADER_RE = re.compile(r"^#{1,3} ")


@dataclass
class KBDoc:
    doc_id: str
    title: str
    source: str
    chunks: list[str]


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Разделяет markdown на dict метаданных (YAML между --- маркерами) и тело."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            meta = yaml.safe_load("\n".join(lines[1:i])) or {}
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return meta, body
    return {}, text


def _split_sections(body: str) -> list[str]:
    """Режет текст на разделы по заголовкам ^#{1,3}; заголовок остаётся в своём разделе."""
    sections: list[str] = []
    current: list[str] = []
    for line in body.split("\n"):
        if _HEADER_RE.match(line) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))
    return [s for s in sections if s.strip()]


def _window_cut(section: str, max_chars: int, overlap: int) -> list[str]:
    """Режет длинный раздел скользящим окном max_chars с перекрытием overlap."""
    chunks: list[str] = []
    start = 0
    while start < len(section):
        end = min(start + max_chars, len(section))
        chunks.append(section[start:end])
        if end == len(section):
            break
        start = end - overlap
    return chunks


def chunk_markdown(body: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """
    Heading-aware чанкинг: сплит по заголовкам, склейка соседних мелких
    разделов до max_chars, нарезка длинных разделов окном с overlap.
    """
    chunks: list[str] = []
    current = ""
    for section in _split_sections(body):
        if len(section) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_window_cut(section, max_chars, overlap))
        elif not current:
            current = section
        elif len(current) + len(section) + 1 <= max_chars:
            current = current + "\n" + section
        else:
            chunks.append(current)
            current = section
    if current:
        chunks.append(current)
    return chunks


def iter_kb_documents(kb_dir: Path) -> Iterator[KBDoc]:
    """Читает *.md из kb_dir и отдаёт документы с чанками."""
    for path in sorted(Path(kb_dir).glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        yield KBDoc(
            doc_id=path.stem,
            title=str(meta.get("title", "")),
            source=str(meta.get("source", "")),
            chunks=chunk_markdown(body),
        )


def _batched(items: list, size: int) -> Iterator[list]:
    it = iter(items)
    while batch := list(islice(it, size)):
        yield batch


def build_index(settings: Settings) -> int:
    """
    Перестраивает индекс ChromaDB из базы знаний.
    Возвращает число проиндексированных чанков.
    """
    embeddings = YandexEmbeddings(
        api_key=settings.yandex_api_key,
        folder_id=settings.yandex_folder_id,
    )
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    records = []
    for doc in iter_kb_documents(settings.kb_dir):
        for i, chunk in enumerate(doc.chunks):
            records.append(
                (
                    f"{doc.doc_id}#{i}",
                    chunk,
                    {"source": doc.source, "title": doc.title, "doc_id": doc.doc_id},
                )
            )

    for batch in _batched(records, _BATCH_SIZE):
        ids, texts, metadatas = zip(*batch)
        collection.add(
            ids=list(ids),
            documents=list(texts),
            metadatas=list(metadatas),
            embeddings=embeddings.embed_documents(list(texts)),
        )

    return len(records)
