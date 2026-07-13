"""
Neo4j-backed document store.

User-added documents are persisted to Neo4j as:
  (:Entity:Document)-[:RELATES {relation: "has_chunk"}]->(:Entity:Chunk)
  (:Entity:Chunk)-[:RELATES {relation: "mentions"}]->(:Entity)

The in-memory GraphSAGE graph is rebuilt from these saved documents after a
server restart.
"""

from __future__ import annotations

import hashlib

from config import NEO4J_CONFIGURED, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from document_ingestion import ingest_document


def load_documents() -> list[dict]:
    driver = None
    try:
        driver = _get_driver()
        with driver.session() as session:
            _ensure_schema(session)
            rows = session.run(
                """
                MATCH (d:Document)
                RETURN d.id AS id, d.source AS source, d.text AS text
                ORDER BY d.created_at, d.id
                """
            ).data()
        return [
            {"id": row["id"], "source": row.get("source"), "text": row["text"]}
            for row in rows
            if row.get("text")
        ]
    except Exception as e:
        print(f"[DocumentStore] Neo4j unavailable while loading documents: {e}")
        return []
    finally:
        if driver:
            driver.close()


def save_document(text: str, source: str | None = None) -> dict:
    record = {
        "id": _document_id(text, source),
        "source": source.strip() if source and source.strip() else None,
        "text": text,
    }
    source_name = record["source"] or "貼り付け文書"
    graph = ingest_document([], [], text, source_name)

    driver = _get_driver()
    try:
        with driver.session() as session:
            _ensure_schema(session)
            session.execute_write(_write_document_graph, record, graph)
    finally:
        driver.close()

    return record


def apply_saved_documents(
    entities: list[dict],
    relations: list[dict],
) -> tuple[list[dict], list[dict]]:
    for document in load_documents():
        result = ingest_document(
            current_entities=entities,
            current_relations=relations,
            text=document["text"],
            source=document.get("source"),
        )
        entities = result["entities"]
        relations = result["relations"]
    return entities, relations


def _write_document_graph(tx, record: dict, graph: dict) -> None:
    entities = graph["entities"]
    relations = graph["relations"]

    for entity in entities:
        etype = entity.get("type", "concept")
        tx.run(
            """
            MERGE (e:Entity {name: $name})
            SET e.type = $type,
                e.source = coalesce($source, e.source),
                e.text = coalesce($text, e.text),
                e.id = coalesce($id, e.id),
                e.created_at = coalesce(e.created_at, datetime())
            """,
            name=entity["name"],
            type=etype,
            source=entity.get("source") or record.get("source"),
            text=entity.get("text") or (record["text"] if etype == "document" else None),
            id=record["id"] if etype == "document" else None,
        )
        if etype == "document":
            tx.run("MATCH (e:Entity {name: $name}) SET e:Document", name=entity["name"])
        elif etype == "chunk":
            tx.run("MATCH (e:Entity {name: $name}) SET e:Chunk", name=entity["name"])

    for rel in relations:
        tx.run(
            """
            MATCH (s:Entity {name: $source})
            MATCH (t:Entity {name: $target})
            MERGE (s)-[r:RELATES {relation: $relation}]->(t)
            SET r.source = "document_ingestion"
            """,
            source=rel["source"],
            target=rel["target"],
            relation=rel.get("relation", ""),
        )


def _ensure_schema(session) -> None:
    session.run(
        """
        CREATE CONSTRAINT entity_name IF NOT EXISTS
        FOR (e:Entity) REQUIRE e.name IS UNIQUE
        """
    )
    session.run(
        """
        CREATE CONSTRAINT document_id IF NOT EXISTS
        FOR (d:Document) REQUIRE d.id IS UNIQUE
        """
    )


def _get_driver():
    if not NEO4J_CONFIGURED:
        raise RuntimeError(
            "NEO4J_PASSWORD is not configured. Create .env and set NEO4J_URI, "
            "NEO4J_USER, and NEO4J_PASSWORD before saving documents to Neo4j."
        )
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise RuntimeError(
            "neo4j package is not installed. Run: pip install neo4j"
        ) from e
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as e:
        driver.close()
        raise RuntimeError(
            f"Neo4jに接続できません。Neo4j DesktopでDBMSがStartしているか、"
            f"Boltポート7687が有効か確認してください。URI={NEO4J_URI}. 詳細: {e}"
        ) from e
    return driver


def _document_id(text: str, source: str | None = None) -> str:
    source_name = source.strip() if source and source.strip() else ""
    return hashlib.sha1(f"{source_name}\n{text}".encode("utf-8")).hexdigest()
