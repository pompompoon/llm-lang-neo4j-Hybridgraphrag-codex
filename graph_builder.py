"""
知識グラフ → PyTorch Geometric Data 変換
- 合成データ or Neo4j からエンティティ・リレーションを読み込み
- ノード特徴量を生成（タイプ埋め込み or テキスト埋め込み）
- PyG の Data オブジェクトに変換
"""

import torch
import numpy as np
from torch_geometric.data import Data

from sample_data import ENTITY_TYPES


def build_pyg_data(entities, relations, features=None, name_to_idx=None):
    """
    エンティティ・リレーション → PyG Data に変換

    Args:
        entities: [{"name": ..., "type": ...}, ...]
        relations: [{"source": ..., "target": ..., "relation": ...}, ...]
        features: (任意) ノード特徴量テンソル [N, D]
        name_to_idx: (任意) {name: idx} マッピング

    Returns:
        data: PyG Data
        name_to_idx: {name: idx}
        idx_to_name: {idx: name}
    """
    if name_to_idx is None:
        name_to_idx = {e["name"]: i for i, e in enumerate(entities)}

    idx_to_name = {v: k for k, v in name_to_idx.items()}
    n_nodes = len(entities)

    # ノード特徴量
    if features is not None:
        x = features if isinstance(features, torch.Tensor) else torch.tensor(features, dtype=torch.float)
    else:
        x = _generate_type_features(entities, n_nodes)

    # エッジ（無向化: 双方向に追加）
    src_list, dst_list = [], []
    for rel in relations:
        s = name_to_idx.get(rel["source"])
        t = name_to_idx.get(rel["target"])
        if s is not None and t is not None:
            src_list.extend([s, t])
            dst_list.extend([t, s])

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = n_nodes

    return data, name_to_idx, idx_to_name


def _generate_type_features(entities, n_nodes, feat_dim=16):
    """タイプごとの疑似特徴量を生成（合成データ用）"""
    np.random.seed(42)
    type_vectors = {}
    for t_name, t_id in ENTITY_TYPES.items():
        base = np.zeros(feat_dim)
        base[t_id * 3: t_id * 3 + 3] = 1.0
        type_vectors[t_name] = base

    features = []
    for ent in entities:
        t = ent.get("type", "concept")
        base = type_vectors.get(t, np.zeros(feat_dim))
        noise = np.random.randn(feat_dim) * 0.1
        features.append(base + noise)

    return torch.tensor(np.array(features), dtype=torch.float)


def load_graph_from_neo4j(uri, user, password):
    """
    Neo4j（第2世代 GraphRAG）からエンティティ・リレーションを読み込み

    Returns:
        entities, relations, features, name_to_idx
    """
    from neo4j_loader import load_from_neo4j
    return load_from_neo4j(uri, user, password)


def build_from_sample(use_new=False):
    """
    合成データから PyG Data を構築

    Args:
        use_new: True なら BASE+NEW（25エンティティ）
    """
    from sample_data import build_base_graph, build_full_graph

    if use_new:
        entities, relations = build_full_graph()
    else:
        entities, relations = build_base_graph()

    data, name_to_idx, idx_to_name = build_pyg_data(entities, relations)
    return data, entities, relations, name_to_idx, idx_to_name
