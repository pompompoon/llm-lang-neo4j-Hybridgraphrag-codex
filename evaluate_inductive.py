"""
Inductive 性の実証スクリプト
BASE グラフ（20エンティティ）で学習 → NEW エンティティ（5個）を追加して
再学習なしに埋め込み生成、までを一気に実行
"""

import time
from train import train_graphsage
from inductive_engine import InductiveEngine
from sample_data import build_full_graph, NEW_ENTITIES
from graph_builder import build_pyg_data


def evaluate():
    print("=" * 60)
    print("GraphSAGE Inductive 性の実証")
    print("=" * 60)

    # Phase 1: BASE グラフ（20エンティティ）で学習
    print("\n[Phase 1] BASE グラフ（20エンティティ）で学習")
    model, data, name_to_idx, idx_to_name, entities, relations, metrics = train_graphsage(
        data_source="sample", epochs=200
    )
    print(f"  学習時間: {metrics['train_time_sec']}秒")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  エンティティ数: {metrics['num_entities']}")

    engine = InductiveEngine()
    engine.set_model(model, data, name_to_idx, idx_to_name, entities, relations)

    # BASE の埋め込み
    z_base = engine.encode_all()
    print(f"  埋め込み形状: {z_base.shape}")

    # Phase 2: NEW エンティティ（5個）を追加 → 再学習なしに推論
    print("\n[Phase 2] NEW エンティティ（5個）を追加 → 再学習なしに推論")
    all_entities, all_relations = build_full_graph()
    new_data, new_n2i, new_i2n = build_pyg_data(all_entities, all_relations)

    start = time.time()
    engine.update_graph(all_entities, all_relations)
    z_full = engine.encode_all()
    infer_time = time.time() - start

    print(f"  推論時間: {infer_time:.4f}秒 （再学習なし）")
    print(f"  埋め込み形状: {z_full.shape}")

    # Phase 3: 新規エンティティの類似検索
    print("\n[Phase 3] 新規エンティティの類似検索")
    for new_ent in NEW_ENTITIES:
        name = new_ent["name"]
        similar = engine.find_similar_by_structure(name, top_k=3)
        print(f"\n  {name} ({new_ent['type']}) の構造的に類似するエンティティ:")
        for s in similar:
            marker = " [NEW]" if s["is_new"] else ""
            print(f"    - {s['name']} ({s['type']}): {s['score']:.4f}{marker}")

    # Phase 4: リンク予測
    print("\n[Phase 4] リンク予測")
    test_pairs = [
        ("Ilya Sutskever", "Sam Altman"),
        ("Ilya Sutskever", "Yann LeCun"),
        ("Claude", "GPT"),
        ("GraphSAGE", "Transformer"),
    ]
    for a, b in test_pairs:
        result = engine.predict_link(a, b)
        exists = "✓ 既存" if result.get("existing_edge") else "✗ 未接続"
        print(f"  {a} ↔ {b}: {result['link_probability']:.4f} ({exists})")

    # Phase 5: ハイブリッド検索
    print("\n[Phase 5] ハイブリッド検索")
    queries = ["GNN", "AI安全性", "OpenAI"]
    for q in queries:
        results = engine.hybrid_search(q, top_k=3)
        print(f"\n  クエリ: '{q}'")
        for r in results:
            marker = " [NEW]" if r["is_new"] else ""
            print(f"    - {r['name']}: score={r['score']:.4f}{marker}")

    print("\n" + "=" * 60)
    print("実証完了")
    print(f"  BASE: {metrics['num_entities']}エンティティで学習")
    print(f"  NEW: {len(NEW_ENTITIES)}エンティティを追加")
    print(f"  再学習なしの推論時間: {infer_time:.4f}秒")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
