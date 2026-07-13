"""
Inductive 推論エンジン
学習済み GraphSAGE モデルで:
  - 全エンティティの構造埋め込み生成（新規含む）
  - 構造ベース類似検索
  - テキスト+構造のハイブリッド検索
  - リンク予測によるグラフ補完
"""

import torch
import torch.nn.functional as F
import numpy as np
import re
from typing import Optional

from graphsage_model import GraphSAGELinkPredictor
from graph_builder import build_pyg_data


class InductiveEngine:
    def __init__(self):
        self.model: Optional[GraphSAGELinkPredictor] = None
        self.data = None
        self.entities = []
        self.relations = []
        self.name_to_idx = {}
        self.idx_to_name = {}
        self.structure_embeddings: Optional[torch.Tensor] = None
        self.text_embeddings: Optional[torch.Tensor] = None
        self.base_entity_names: set = set()
        self.model_type = "graphsage"
        self.is_trained = False

    def set_model(
        self,
        model,
        data,
        name_to_idx,
        idx_to_name,
        entities,
        relations,
        model_type: str = "graphsage",
    ):
        """学習済みモデルとデータを設定"""
        self.model = model
        self.data = data
        self.name_to_idx = name_to_idx
        self.idx_to_name = idx_to_name
        self.entities = entities
        self.relations = relations
        self.base_entity_names = set(name_to_idx.keys())
        self.model_type = (model_type or "graphsage").lower()
        self.is_trained = True

    def update_graph(self, entities, relations, features=None):
        """
        グラフを更新（新規エンティティ追加）
        学習済みモデルは再学習せずそのまま使う（inductive）
        """
        name_to_idx = {e["name"]: i for i, e in enumerate(entities)}
        data, name_to_idx, idx_to_name = build_pyg_data(
            entities, relations, features=features, name_to_idx=name_to_idx
        )
        self.data = data
        self.entities = entities
        self.relations = relations
        self.name_to_idx = name_to_idx
        self.idx_to_name = idx_to_name
        self.structure_embeddings = None  # 再計算が必要

    def encode_all(self) -> torch.Tensor:
        """全エンティティの構造埋め込みを生成（inductive推論）"""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train first.")

        self.model.eval()
        with torch.no_grad():
            self.structure_embeddings = self.model.encode(
                self.data.x, self.data.edge_index
            )

        return self.structure_embeddings

    def find_similar_by_structure(self, entity_name: str, top_k: int = 5) -> list[dict]:
        """構造ベースの類似エンティティ検索"""
        if self.structure_embeddings is None:
            self.encode_all()

        idx = self.name_to_idx.get(entity_name)
        if idx is None:
            return []

        query = self.structure_embeddings[idx].unsqueeze(0)
        sims = F.cosine_similarity(query, self.structure_embeddings, dim=1)
        sims[idx] = -1  # 自分自身を除外

        top_indices = sims.argsort(descending=True)[:top_k]
        results = []
        for i in top_indices:
            i = i.item()
            name = self.idx_to_name.get(i, f"node_{i}")
            results.append({
                "name": name,
                "type": self._get_type(name),
                "score": round(sims[i].item(), 4),
                "is_new": name not in self.base_entity_names,
            })

        return results

    def hybrid_search(self, query: str, top_k: int = 5, text_weight: float = 0.5) -> list[dict]:
        """
        ハイブリッド検索: テキスト埋め込み + 構造埋め込み

        Args:
            query: 検索クエリ
            top_k: 返す件数
            text_weight: テキスト埋め込みの重み (0.0=構造のみ, 1.0=テキストのみ)
        """
        if self.structure_embeddings is None:
            self.encode_all()

        text_weight = max(0.0, min(1.0, float(text_weight)))
        struct_weight = 1.0 - text_weight

        # Text matching finds query anchor nodes. GraphSAGE embeddings then
        # expand those anchors to structurally similar nodes.
        text_scores = self._text_match_scores(query)
        struct_scores = self._structure_scores(text_scores)

        combined = text_weight * text_scores + struct_weight * struct_scores

        top_indices = combined.argsort(descending=True)[:top_k]
        results = []
        for i in top_indices:
            i = i.item()
            name = self.idx_to_name.get(i, f"node_{i}")
            score = combined[i].item()
            if score > 0:
                results.append({
                    "name": name,
                    "type": self._get_type(name),
                    "score": round(score, 4),
                    "text_score": round(text_scores[i].item(), 4),
                    "struct_score": round(struct_scores[i].item(), 4),
                    "is_new": name not in self.base_entity_names,
                })

        return results

    def suggest_missing_links(self, entity_name: str, top_k: int = 5) -> list[dict]:
        """リンク予測によるグラフ補完の提案"""
        if self.structure_embeddings is None:
            self.encode_all()

        idx = self.name_to_idx.get(entity_name)
        if idx is None:
            return []

        z = self.structure_embeddings
        existing = set()
        edge_index = self.data.edge_index
        for j in range(edge_index.size(1)):
            s, t = edge_index[0, j].item(), edge_index[1, j].item()
            if s == idx:
                existing.add(t)
            if t == idx:
                existing.add(s)

        candidates = []
        for j in range(len(self.entities)):
            if j == idx or j in existing:
                continue
            score = (z[idx] * z[j]).sum().sigmoid().item()
            name = self.idx_to_name.get(j, f"node_{j}")
            candidates.append({
                "name": name,
                "type": self._get_type(name),
                "link_probability": round(score, 4),
                "is_new": name not in self.base_entity_names,
            })

        candidates.sort(key=lambda x: x["link_probability"], reverse=True)
        return candidates[:top_k]

    def predict_link(self, entity_a: str, entity_b: str) -> dict:
        """2エンティティ間のリンク確率を予測"""
        if self.structure_embeddings is None:
            self.encode_all()

        idx_a = self.name_to_idx.get(entity_a)
        idx_b = self.name_to_idx.get(entity_b)

        if idx_a is None or idx_b is None:
            return {"error": f"Entity not found: {entity_a if idx_a is None else entity_b}"}

        z = self.structure_embeddings
        score = (z[idx_a] * z[idx_b]).sum().sigmoid().item()

        # 既存エッジかチェック
        edge_index = self.data.edge_index
        exists = False
        for j in range(edge_index.size(1)):
            if (edge_index[0, j] == idx_a and edge_index[1, j] == idx_b) or \
               (edge_index[0, j] == idx_b and edge_index[1, j] == idx_a):
                exists = True
                break

        return {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "link_probability": round(score, 4),
            "existing_edge": exists,
        }

    def get_graph_data(self) -> dict:
        """グラフ可視化データを返す"""
        nodes = []
        type_counts = {}
        for i, ent in enumerate(self.entities):
            etype = ent.get("type", "concept")
            type_counts[etype] = type_counts.get(etype, 0) + 1
            nodes.append({
                "id": ent["name"],
                "label": ent["name"],
                "type": etype,
                "is_new": ent["name"] not in self.base_entity_names,
            })

        edges = [
            {"source": r["source"], "target": r["target"], "relation": r.get("relation", "")}
            for r in self.relations
        ]

        return {
            "nodes": nodes,
            "edges": edges,
            "type_counts": type_counts,
            "stats": {
                "nodes": len(nodes),
                "edges": len(edges),
                "communities": len(type_counts),
                "chunks": type_counts.get("chunk", 0),
                "documents": type_counts.get("document", 0),
                "base_nodes": len(self.base_entity_names),
                "new_nodes": len(nodes) - len(self.base_entity_names),
                "model_type": self.model_type,
            },
        }

    def _text_match_scores(self, query: str) -> torch.Tensor:
        """クエリ文字列とのテキストマッチスコア（簡易実装）"""
        scores = torch.zeros(len(self.entities))
        query_lower = query.lower()
        query_terms = [
            term for term in re.split(r"[\s、。・はをにがのとでへもやか?？]+", query_lower)
            if len(term) >= 2
        ]
        if query_lower and query_lower not in query_terms:
            query_terms.append(query_lower)

        for i, ent in enumerate(self.entities):
            name_lower = ent["name"].lower()
            text_lower = ent.get("text", "").lower()
            source_lower = ent.get("source", "").lower()
            searchable = " ".join(part for part in [name_lower, text_lower, source_lower] if part)
            # 完全一致
            if query_lower == name_lower:
                scores[i] = 1.0
            # 部分一致
            elif query_lower in searchable or name_lower in query_lower:
                scores[i] = 0.8
            # 単語単位の部分一致
            else:
                for term in query_terms:
                    if term in searchable or name_lower in term:
                        scores[i] = max(scores[i], 0.5)

            # 構造埋め込みの類似度も加味
            if self.structure_embeddings is not None and scores[i] > 0:
                # マッチしたノードの構造的な近傍も少しスコアを付ける
                pass

        return scores

    def _structure_scores(self, anchor_scores: torch.Tensor) -> torch.Tensor:
        """Score nodes by GraphSAGE proximity to text-matched anchor nodes."""
        if self.structure_embeddings is None or anchor_scores.numel() == 0:
            return torch.zeros_like(anchor_scores)

        anchor_mask = anchor_scores > 0
        if not torch.any(anchor_mask):
            return torch.zeros_like(anchor_scores)

        embeddings = F.normalize(self.structure_embeddings, p=2, dim=1)
        device_mask = anchor_mask.to(embeddings.device)
        weights = anchor_scores[anchor_mask].to(embeddings.device)
        anchor_embeddings = embeddings[device_mask]
        centroid = (anchor_embeddings * weights.unsqueeze(1)).sum(dim=0)
        centroid = centroid / weights.sum().clamp_min(1e-8)
        centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1)

        scores = F.cosine_similarity(
            embeddings,
            centroid.expand_as(embeddings),
            dim=1,
        )
        return scores.clamp(min=0.0, max=1.0).cpu()

    def _get_type(self, name: str) -> str:
        for ent in self.entities:
            if ent["name"] == name:
                return ent.get("type", "concept")
        return "concept"


# シングルトン
_engine: Optional[InductiveEngine] = None


def get_engine() -> InductiveEngine:
    global _engine
    if _engine is None:
        _engine = InductiveEngine()
    return _engine
