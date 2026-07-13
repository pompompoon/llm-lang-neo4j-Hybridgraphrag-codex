"""
FastAPI サーバー
LLMGraphRAG の全エンドポイントを提供
"""

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_provider_info


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
if not STATIC_DIR.exists():
    STATIC_DIR = BASE_DIR

app = FastAPI(title="LLMGraphRAG", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def load_persisted_documents():
    from document_store import apply_saved_documents, load_documents
    from inductive_engine import get_engine
    from sample_data import build_full_graph

    if not load_documents():
        return

    entities, relations = build_full_graph()
    entities, relations = apply_saved_documents(entities, relations)
    get_engine().update_graph(entities, relations)


@app.get("/")
async def root():
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/health")
async def health():
    from inductive_engine import get_engine

    engine = get_engine()
    return {
        "status": "ok",
        "trained": engine.is_trained,
        "entities": len(engine.entities),
        "model_type": engine.model_type,
        **get_provider_info(),
    }


@app.get("/api/documents/status")
async def document_status():
    from document_store import load_documents

    documents = load_documents()
    return {
        "store": "neo4j",
        "documents": len(documents),
        "sources": [doc.get("source") for doc in documents[-10:]],
    }


# --- GraphSAGE 学習・推論 ---

class TrainRequest(BaseModel):
    data_source: str = "sample"
    model_type: str = "graphsage"
    epochs: int = 200
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"


@app.post("/api/graphsage/train")
async def train_model(req: TrainRequest):
    from document_store import apply_saved_documents
    from inductive_engine import get_engine
    from train import train_graphsage

    try:
        model, data, n2i, i2n, ents, rels, metrics = await asyncio.to_thread(
            train_graphsage,
            data_source=req.data_source,
            model_type=req.model_type,
            epochs=req.epochs,
            neo4j_uri=req.neo4j_uri,
            neo4j_user=req.neo4j_user,
            neo4j_password=req.neo4j_password,
        )
        engine = get_engine()
        engine.set_model(model, data, n2i, i2n, ents, rels, model_type=req.model_type)

        # Keep persisted documents visible immediately after retraining.
        entities, relations = apply_saved_documents(engine.entities, engine.relations)
        engine.update_graph(entities, relations)
        z = engine.encode_all()

        return {
            "status": "trained",
            "shape": list(z.shape),
            **metrics,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/graphsage/embed")
async def embed_all():
    from document_store import apply_saved_documents
    from inductive_engine import get_engine

    engine = get_engine()
    if not engine.is_trained:
        raise HTTPException(400, "Model not trained")

    # Do not replace a Neo4j-trained graph with the sample graph.
    entities, relations = apply_saved_documents(engine.entities, engine.relations)
    engine.update_graph(entities, relations)
    z = engine.encode_all()

    return {
        "status": "embedded",
        "shape": list(z.shape),
        "entities": len(entities),
    }


# --- チャット ---

class ChatRequest(BaseModel):
    message: str


def _run_agent(question: str) -> dict:
    from langgraph_agent import AgentState, build_agent

    agent = build_agent()
    initial: AgentState = {
        "question": question,
        "search_results": [],
        "context": "",
        "answer": "",
        "quality_score": 0.0,
        "retry_count": 0,
        "search_query": "",
        "metrics": {},
    }
    return agent.invoke(initial)


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        result = await asyncio.to_thread(_run_agent, req.message)
        return {
            "answer": result.get("answer", ""),
            "quality_score": result.get("quality_score", 0),
            "metrics": result.get("metrics", {}),
            "retry_count": result.get("retry_count", 0),
            "search_results": result.get("search_results", []),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# --- 一括評価 ---

class EvalItem(BaseModel):
    question: str
    expected: Optional[str] = None


class EvaluateRequest(BaseModel):
    items: list[EvalItem]


@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    """複数の質問（+任意の期待回答）でRAGの精度指標を一括算出"""
    from rag_metrics import expected_match

    if not req.items:
        raise HTTPException(400, "items is empty")

    results = []
    for item in req.items:
        try:
            r = await asyncio.to_thread(_run_agent, item.question)
            metrics = r.get("metrics", {})
            entry = {
                "question": item.question,
                "answer": r.get("answer", ""),
                "metrics": metrics,
                "retry_count": r.get("retry_count", 0),
                "hits": len(r.get("search_results", [])),
            }
            if item.expected:
                entry["expected_match"] = expected_match(r.get("answer", ""), item.expected)
            results.append(entry)
        except Exception as e:
            results.append({"question": item.question, "error": str(e)})

    # 集計（エラー分を除く）
    ok = [r for r in results if "metrics" in r]
    summary: dict = {"total": len(results), "succeeded": len(ok), "failed": len(results) - len(ok)}
    if ok:
        keys = ["faithfulness", "answer_relevancy", "context_relevancy",
                "grounded_numbers", "overall", "quality_score", "llm_overall"]
        for key in keys:
            vals = [r["metrics"][key] for r in ok if key in r["metrics"]]
            if vals:
                summary[f"avg_{key}"] = round(sum(vals) / len(vals), 3)
        em = [r["expected_match"] for r in ok if "expected_match" in r]
        if em:
            summary["avg_expected_match"] = round(sum(em) / len(em), 3)

    return {"summary": summary, "results": results}


# --- 検索 ---

class HybridSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    text_weight: float = 0.5


@app.post("/api/search/hybrid")
async def hybrid_search(req: HybridSearchRequest):
    from inductive_engine import get_engine

    engine = get_engine()
    if not engine.is_trained:
        raise HTTPException(400, "Model not trained")
    results = engine.hybrid_search(req.query, req.top_k, req.text_weight)
    return {"query": req.query, "results": results}


@app.get("/api/entities/{name}/similar")
async def similar_entities(name: str, top_k: int = 5):
    from inductive_engine import get_engine

    engine = get_engine()
    if not engine.is_trained:
        raise HTTPException(400, "Model not trained")
    results = engine.find_similar_by_structure(name, top_k)
    return {"entity": name, "similar": results}


# --- ドキュメント追加 ---

class DocumentAddRequest(BaseModel):
    text: str
    source: Optional[str] = None
    data_source: str = "sample"
    model_type: str = "graphsage"


@app.post("/api/documents")
@app.post("/api/documents/add")
async def add_document(req: DocumentAddRequest):
    from document_store import apply_saved_documents, save_document
    from document_ingestion import ingest_document
    from inductive_engine import get_engine
    from sample_data import build_full_graph

    engine = get_engine()
    auto_trained = False

    try:
        if not engine.is_trained:
            from train import train_graphsage

            model, data, n2i, i2n, ents, rels, _metrics = await asyncio.to_thread(
                train_graphsage,
                data_source=req.data_source,
                model_type=req.model_type,
                epochs=50,
            )
            engine.set_model(model, data, n2i, i2n, ents, rels, model_type=req.model_type)
            auto_trained = True

        current_entities = engine.entities
        current_relations = engine.relations

        if not current_entities:
            current_entities, current_relations = build_full_graph()

        if auto_trained:
            current_entities, current_relations = apply_saved_documents(
                current_entities, current_relations
            )

        result = ingest_document(
            current_entities=current_entities,
            current_relations=current_relations,
            text=req.text,
            source=req.source,
        )
        saved = save_document(req.text, req.source)
        engine.update_graph(result["entities"], result["relations"])
        z = engine.encode_all()

        return {
            "status": "added",
            "auto_trained": auto_trained,
            "persisted_to": "neo4j",
            "document_id": saved["id"],
            **result["summary"],
            "shape": list(z.shape),
            "stats": engine.get_graph_data().get("stats", {}),
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# --- リンク予測 ---

class LinkPredictRequest(BaseModel):
    entity_a: str
    entity_b: str


@app.post("/api/links/predict")
async def predict_link(req: LinkPredictRequest):
    from inductive_engine import get_engine

    engine = get_engine()
    if not engine.is_trained:
        raise HTTPException(400, "Model not trained")
    return engine.predict_link(req.entity_a, req.entity_b)


@app.get("/api/links/suggest/{name}")
async def suggest_links(name: str, top_k: int = 5):
    from inductive_engine import get_engine

    engine = get_engine()
    if not engine.is_trained:
        raise HTTPException(400, "Model not trained")
    return {"entity": name, "suggestions": engine.suggest_missing_links(name, top_k)}


# --- グラフ ---

@app.get("/api/graph/stats")
async def graph_stats():
    from inductive_engine import get_engine

    engine = get_engine()
    gd = engine.get_graph_data()
    return gd.get("stats", {})


@app.get("/api/graph/data")
async def graph_data():
    from inductive_engine import get_engine

    engine = get_engine()
    return engine.get_graph_data()


# --- Neo4j 書き戻し ---

class WritebackRequest(BaseModel):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    property_name: str = "struct_embedding"


@app.post("/api/neo4j/writeback")
async def neo4j_writeback(req: WritebackRequest):
    from inductive_engine import get_engine
    from neo4j_loader import write_structure_embeddings

    engine = get_engine()
    if engine.structure_embeddings is None:
        raise HTTPException(400, "No embeddings. Run /api/graphsage/embed first.")

    try:
        await asyncio.to_thread(
            write_structure_embeddings,
            req.neo4j_uri,
            req.neo4j_user,
            req.neo4j_password,
            engine.name_to_idx,
            engine.structure_embeddings,
            req.property_name,
        )
        return {
            "status": "written",
            "property": req.property_name,
            "entities": len(engine.name_to_idx),
        }
    except Exception as e:
        raise HTTPException(500, str(e))
