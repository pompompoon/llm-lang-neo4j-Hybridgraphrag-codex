"""
Neo4j ローダー
第2世代 GraphRAG の Neo4j からエンティティ・リレーションを読み書き
遅延インポート設計: neo4j パッケージ未インストールでも他機能は動作
"""

import numpy as np
import torch


def load_from_neo4j(uri: str, user: str, password: str):
    """
    Neo4j からエンティティ・リレーションを読み込み

    Returns:
        entities, relations, features (torch.Tensor), name_to_idx
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # エンティティ取得
        node_records = session.run("""
            MATCH (e:Entity)
            RETURN e.name AS name, e.type AS type, e.embedding AS embedding
            ORDER BY e.name
        """).data()

        # リレーション取得
        edge_records = session.run("""
            MATCH (a:Entity)-[r:RELATES]->(b:Entity)
            RETURN a.name AS source, b.name AS target, r.relation AS relation
        """).data()

    driver.close()

    entities = [
        {"name": r["name"], "type": r["type"] or "concept"}
        for r in node_records
    ]
    relations = [
        {"source": r["source"], "target": r["target"], "relation": r["relation"] or ""}
        for r in edge_records
    ]

    name_to_idx = {e["name"]: i for i, e in enumerate(entities)}

    # ノード特徴量: テキスト埋め込み（Entity.embedding）があればそれを使用
    features = None
    has_embeddings = all(r.get("embedding") is not None for r in node_records)
    if has_embeddings and node_records:
        feat_list = [np.array(r["embedding"], dtype=np.float32) for r in node_records]
        features = torch.tensor(np.stack(feat_list), dtype=torch.float)

    print(f"[Neo4jLoader] Loaded {len(entities)} entities, {len(relations)} relations"
          f" (embeddings: {'yes' if features is not None else 'no'})")

    return entities, relations, features, name_to_idx


def write_structure_embeddings(
    uri: str, user: str, password: str,
    name_to_idx: dict, embeddings: torch.Tensor,
    property_name: str = "struct_embedding"
):
    """
    GraphSAGE の構造埋め込みを Neo4j の Entity ノードに書き戻す

    Args:
        name_to_idx: {entity_name: index}
        embeddings: [N, D] テンソル
        property_name: Neo4j プロパティ名
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    emb_np = embeddings.detach().cpu().numpy()
    dim = emb_np.shape[1]

    with driver.session() as session:
        # プロパティを書き込み
        count = 0
        for name, idx in name_to_idx.items():
            vec = emb_np[idx].tolist()
            session.run(f"""
                MATCH (e:Entity {{name: $name}})
                SET e.{property_name} = $vec
            """, name=name, vec=vec)
            count += 1

        # ベクトルインデックスを作成（Neo4j 5.11+）
        index_name = f"entity_{property_name}_index"
        try:
            session.run(f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (e:Entity) ON (e.{property_name})
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: $dim,
                    `vector.similarity_function`: 'cosine'
                }}}}
            """, dim=dim)
            print(f"[Neo4jLoader] Created vector index '{index_name}' (dim={dim})")
        except Exception as e:
            print(f"[Neo4jLoader] Vector index creation: {e}")

    driver.close()
    print(f"[Neo4jLoader] Wrote {count} structure embeddings to Neo4j ({property_name})")
