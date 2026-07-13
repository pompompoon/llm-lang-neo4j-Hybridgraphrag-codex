"""
Simple document ingestion for the demo GraphRAG graph.

The goal here is intentionally modest: turn pasted text into chunk nodes,
lightweight entity nodes, and relations that the existing in-memory
InductiveEngine can embed without retraining the GraphSAGE model.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Iterable


CHUNK_SIZE = 420
CHUNK_OVERLAP = 60
MAX_ENTITIES_PER_CHUNK = 8

STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "between",
    "from",
    "into",
    "that",
    "the",
    "their",
    "then",
    "there",
    "this",
    "with",
    "これは",
    "ため",
    "です",
    "ます",
    "する",
    "した",
    "して",
    "から",
    "こと",
    "この",
    "その",
    "また",
}


def ingest_document(
    current_entities: list[dict],
    current_relations: list[dict],
    text: str,
    source: str | None = None,
) -> dict:
    """Return updated graph lists plus a small ingestion summary."""
    clean_text = _normalize_text(text)
    if not clean_text:
        raise ValueError("Document text is empty.")

    source_name = source.strip() if source and source.strip() else "貼り付け文書"
    doc_hash = hashlib.sha1(f"{source_name}\n{clean_text}".encode("utf-8")).hexdigest()[:10]
    doc_name = f"Document: {source_name} ({doc_hash})"

    entities = _dedupe_entities([*current_entities, {"name": doc_name, "type": "document"}])
    relations = _dedupe_relations(current_relations)
    existing_names = {entity["name"] for entity in entities}

    chunks = _chunk_text(clean_text)
    extracted_names: set[str] = set()
    chunk_names = []

    for index, chunk in enumerate(chunks, start=1):
        # source alone is not unique: two documents from the same source must
        # not MERGE into the same Neo4j chunk node.
        chunk_name = f"{source_name} ({doc_hash}) chunk {index}"
        if chunk_name not in existing_names:
            entities.append({
                "name": chunk_name,
                "type": "chunk",
                "text": chunk,
                "source": source_name,
            })
            existing_names.add(chunk_name)
        chunk_names.append(chunk_name)
        relations.append({"source": doc_name, "target": chunk_name, "relation": "has_chunk"})

        for entity_name in _extract_entities(chunk):
            extracted_names.add(entity_name)
            if entity_name not in existing_names:
                entities.append({"name": entity_name, "type": _guess_entity_type(entity_name)})
                existing_names.add(entity_name)
            relations.append({"source": chunk_name, "target": entity_name, "relation": "mentions"})

    relations = _dedupe_relations(relations)

    return {
        "entities": entities,
        "relations": relations,
        "summary": {
            "document": doc_name,
            "source": source_name,
            "chunks_added": len(chunk_names),
            "entities_extracted": len(extracted_names),
            "relations_added": max(0, len(relations) - len(current_relations)),
        },
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _chunk_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        if end < len(text):
            boundary = max(text.rfind("。", start, end), text.rfind(".", start, end))
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - CHUNK_OVERLAP)

    return [chunk for chunk in chunks if chunk]


def _extract_entities(text: str) -> list[str]:
    candidates = []

    # English/proper-noun style names: OpenAI, GraphSAGE, Sam Altman, Neo4j, etc.
    candidates.extend(
        match.group(0).strip()
        for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9]*(?:[- ][A-Z][A-Za-z0-9]*){0,3}|[A-Za-z]+[A-Z][A-Za-z0-9]*)\b", text)
    )

    # Japanese terms commonly used as compact concepts.
    candidates.extend(
        match.group(0).strip()
        for match in re.finditer(r"[一-龥ぁ-んァ-ン]{3,12}", text)
    )

    filtered = [
        candidate
        for candidate in candidates
        if len(candidate) >= 3 and candidate.lower() not in STOPWORDS
    ]

    counts = Counter(filtered)
    return [name for name, _ in counts.most_common(MAX_ENTITIES_PER_CHUNK)]


def _guess_entity_type(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("inc", "corp", "company", "openai", "google", "anthropic", "meta")):
        return "organization"
    if any(token in lowered for token in ("ai", "graph", "sage", "gpt", "llm", "neo4j", "rag")):
        return "technology"
    return "concept"


def _dedupe_entities(entities: Iterable[dict]) -> list[dict]:
    by_name = {}
    for entity in entities:
        name = entity.get("name")
        if name and name not in by_name:
            by_name[name] = entity
    return list(by_name.values())


def _dedupe_relations(relations: Iterable[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for relation in relations:
        key = (
            relation.get("source"),
            relation.get("target"),
            relation.get("relation", ""),
        )
        if key[0] and key[1] and key not in seen:
            deduped.append(relation)
            seen.add(key)
    return deduped
