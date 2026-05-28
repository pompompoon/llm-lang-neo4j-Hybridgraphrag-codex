"""
合成知識グラフを第2世代 GraphRAG の形式で Neo4j に投入
第2世代 GraphRAG なしでも Neo4j 連携を動作実証できる
"""

import argparse


def seed_neo4j(uri: str, user: str, password: str, wipe: bool = False):
    from neo4j import GraphDatabase
    from sample_data import build_full_graph

    entities, relations = build_full_graph()
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        if wipe:
            session.run("MATCH (e:Entity) DETACH DELETE e")
            print("[SeedNeo4j] Wiped existing Entity nodes")

        # エンティティ投入
        for ent in entities:
            session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $type
            """, name=ent["name"], type=ent["type"])

        # リレーション投入
        for rel in relations:
            session.run("""
                MATCH (a:Entity {name: $src}), (b:Entity {name: $tgt})
                MERGE (a)-[r:RELATES]->(b)
                SET r.relation = $rel
            """, src=rel["source"], tgt=rel["target"], rel=rel.get("relation", ""))

    driver.close()
    print(f"[SeedNeo4j] Seeded {len(entities)} entities, {len(relations)} relations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Neo4j with synthetic knowledge graph")
    parser.add_argument("--uri", default="neo4j://127.0.0.1:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--wipe", action="store_true", help="Delete existing Entity nodes first")
    args = parser.parse_args()

    seed_neo4j(args.uri, args.user, args.password, args.wipe)
