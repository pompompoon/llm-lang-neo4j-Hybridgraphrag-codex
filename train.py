"""
GraphSAGE 自己教師あり学習（リンク予測）
既存エッジ=正例 / ランダム非エッジ=負例 で学習
"""

import time
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

from graphsage_model import build_link_predictor
from graph_builder import build_pyg_data, build_from_sample


def train_graphsage(
    data_source: str = "sample",
    model_type: str = "graphsage",
    epochs: int = 200,
    lr: float = 0.01,
    hidden_channels: int = 64,
    out_channels: int = 32,
    neo4j_uri: str = None,
    neo4j_user: str = None,
    neo4j_password: str = None,
):
    """
    GraphSAGE を学習

    Args:
        data_source: "sample" or "neo4j"
        epochs: 学習エポック数

    Returns:
        model, data, name_to_idx, idx_to_name, entities, relations, metrics
    """
    model_type = (model_type or "graphsage").lower()
    print(f"[Train] model_type={model_type}, data_source={data_source}, epochs={epochs}")
    start = time.time()

    # データ準備
    if data_source == "neo4j":
        from graph_builder import load_graph_from_neo4j
        entities, relations, features, name_to_idx = load_graph_from_neo4j(
            neo4j_uri, neo4j_user, neo4j_password
        )
        data, name_to_idx, idx_to_name = build_pyg_data(
            entities, relations, features=features, name_to_idx=name_to_idx
        )
    else:
        data, entities, relations, name_to_idx, idx_to_name = build_from_sample(use_new=False)

    in_channels = data.x.shape[1]
    model = build_link_predictor(model_type, in_channels, hidden_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 学習ループ
    model.train()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 負例サンプリング
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1),
        )

        pos_score, neg_score = model(
            data.x, data.edge_index, data.edge_index, neg_edge_index
        )

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score, torch.ones_like(pos_score)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_score, torch.zeros_like(neg_score)
        )
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    elapsed = time.time() - start

    # 評価
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        pos_score = model.predict_link(z, data.edge_index).sigmoid()
        neg_edge = negative_sampling(
            data.edge_index, data.num_nodes, data.edge_index.size(1)
        )
        neg_score = model.predict_link(z, neg_edge).sigmoid()

        labels = torch.cat([
            torch.ones(pos_score.size(0)),
            torch.zeros(neg_score.size(0)),
        ])
        scores = torch.cat([pos_score, neg_score])
        auc = roc_auc_score(labels.numpy(), scores.numpy())

    metrics = {
        "final_loss": losses[-1] if losses else 0,
        "auc": auc,
        "epochs": epochs,
        "train_time_sec": round(elapsed, 3),
        "num_entities": len(entities),
        "num_relations": len(relations),
        "num_edges": data.edge_index.size(1),
        "model_type": model_type,
    }

    print(
        f"[Train] {model_type} done in {elapsed:.3f}s | "
        f"AUC={auc:.4f} | Loss={losses[-1]:.4f}"
    )

    return model, data, name_to_idx, idx_to_name, entities, relations, metrics


if __name__ == "__main__":
    model, data, n2i, i2n, ents, rels, metrics = train_graphsage()
    print(f"\nMetrics: {metrics}")
