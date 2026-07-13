"""
GraphSAGE モデル（SAGEConv + リンク予測）
- 2層 SAGEConv（近傍集約 mean）
- 自己教師あり学習: 既存エッジ=正例 / 非エッジ=負例
- inductive: 学習済みモデルで新規ノードにも埋め込み生成可能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv


class GraphSAGEEncoder(nn.Module):
    """2層 GraphSAGE エンコーダ"""

    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """ノード特徴量 + エッジ → 構造埋め込み"""
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv2(h, edge_index)
        return h


class GATEncoder(nn.Module):
    """2層 GAT エンコーダ"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        heads: int = 4,
    ):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=0.3,
        )
        self.conv2 = GATv2Conv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=0.3,
        )
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """ノード特徴量 + エッジ → attention付き構造埋め込み"""
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv2(h, edge_index)
        return h


class LinkPredictor(nn.Module):
    """リンク予測ヘッド（内積ベース）"""

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """2ノード間のリンク確率を予測"""
        src, dst = edge_index[0], edge_index[1]
        return (z[src] * z[dst]).sum(dim=-1)


class GraphSAGELinkPredictor(nn.Module):
    """GraphSAGE + リンク予測の統合モデル"""

    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 32):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, out_channels)
        self.predictor = LinkPredictor()

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def predict_link(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, edge_index)

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        """学習用: 正例・負例のリンク予測スコアを返す"""
        z = self.encode(x, edge_index)
        pos_score = self.predict_link(z, pos_edge_index)
        neg_score = self.predict_link(z, neg_edge_index)
        return pos_score, neg_score


class GATLinkPredictor(nn.Module):
    """GAT + リンク予測の統合モデル"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        heads: int = 4,
    ):
        super().__init__()
        self.encoder = GATEncoder(in_channels, hidden_channels, out_channels, heads=heads)
        self.predictor = LinkPredictor()

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def predict_link(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, edge_index)

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        """学習用: 正例・負例のリンク予測スコアを返す"""
        z = self.encode(x, edge_index)
        pos_score = self.predict_link(z, pos_edge_index)
        neg_score = self.predict_link(z, neg_edge_index)
        return pos_score, neg_score


def build_link_predictor(
    model_type: str,
    in_channels: int,
    hidden_channels: int = 64,
    out_channels: int = 32,
) -> nn.Module:
    """モデル種別からリンク予測モデルを生成"""
    normalized = (model_type or "graphsage").lower()
    if normalized == "gat":
        return GATLinkPredictor(in_channels, hidden_channels, out_channels)
    if normalized == "graphsage":
        return GraphSAGELinkPredictor(in_channels, hidden_channels, out_channels)
    raise ValueError(f"Unsupported model_type: {model_type}")
