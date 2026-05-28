"""
合成知識グラフ（AI業界）
- BASE: 20エンティティ（学習用）
- NEW: 5エンティティ（inductive推論の実証用）
"""

# エンティティタイプ
ENTITY_TYPES = {
    "person": 0,
    "organization": 1,
    "technology": 2,
    "concept": 3,
    "event": 4,
}

# --- BASE グラフ（20エンティティ）---
BASE_ENTITIES = [
    {"name": "Geoffrey Hinton", "type": "person"},
    {"name": "Yann LeCun", "type": "person"},
    {"name": "Yoshua Bengio", "type": "person"},
    {"name": "Demis Hassabis", "type": "person"},
    {"name": "Sam Altman", "type": "person"},
    {"name": "Google", "type": "organization"},
    {"name": "DeepMind", "type": "organization"},
    {"name": "OpenAI", "type": "organization"},
    {"name": "Meta AI", "type": "organization"},
    {"name": "Anthropic", "type": "organization"},
    {"name": "Transformer", "type": "technology"},
    {"name": "GPT", "type": "technology"},
    {"name": "GNN", "type": "technology"},
    {"name": "CNN", "type": "technology"},
    {"name": "深層学習", "type": "concept"},
    {"name": "自然言語処理", "type": "concept"},
    {"name": "コンピュータビジョン", "type": "concept"},
    {"name": "強化学習", "type": "concept"},
    {"name": "NeurIPS", "type": "event"},
    {"name": "ICML", "type": "event"},
]

BASE_RELATIONS = [
    {"source": "Geoffrey Hinton", "target": "Google", "relation": "worked_at"},
    {"source": "Geoffrey Hinton", "target": "深層学習", "relation": "pioneered"},
    {"source": "Geoffrey Hinton", "target": "CNN", "relation": "developed"},
    {"source": "Yann LeCun", "target": "Meta AI", "relation": "leads"},
    {"source": "Yann LeCun", "target": "CNN", "relation": "invented"},
    {"source": "Yann LeCun", "target": "コンピュータビジョン", "relation": "researches"},
    {"source": "Yoshua Bengio", "target": "深層学習", "relation": "pioneered"},
    {"source": "Yoshua Bengio", "target": "自然言語処理", "relation": "researches"},
    {"source": "Demis Hassabis", "target": "DeepMind", "relation": "founded"},
    {"source": "Demis Hassabis", "target": "強化学習", "relation": "researches"},
    {"source": "Sam Altman", "target": "OpenAI", "relation": "leads"},
    {"source": "Sam Altman", "target": "GPT", "relation": "oversees"},
    {"source": "DeepMind", "target": "Google", "relation": "acquired_by"},
    {"source": "DeepMind", "target": "強化学習", "relation": "specializes"},
    {"source": "OpenAI", "target": "GPT", "relation": "developed"},
    {"source": "OpenAI", "target": "Transformer", "relation": "uses"},
    {"source": "Meta AI", "target": "GNN", "relation": "researches"},
    {"source": "Anthropic", "target": "自然言語処理", "relation": "specializes"},
    {"source": "Anthropic", "target": "OpenAI", "relation": "spun_off_from"},
    {"source": "Transformer", "target": "自然言語処理", "relation": "revolutionized"},
    {"source": "Transformer", "target": "GPT", "relation": "basis_of"},
    {"source": "GNN", "target": "深層学習", "relation": "subfield_of"},
    {"source": "CNN", "target": "深層学習", "relation": "subfield_of"},
    {"source": "CNN", "target": "コンピュータビジョン", "relation": "used_in"},
    {"source": "NeurIPS", "target": "深層学習", "relation": "covers"},
    {"source": "ICML", "target": "深層学習", "relation": "covers"},
    {"source": "Geoffrey Hinton", "target": "NeurIPS", "relation": "publishes_at"},
    {"source": "Yann LeCun", "target": "ICML", "relation": "publishes_at"},
]

# --- NEW エンティティ（5個、inductive推論用）---
NEW_ENTITIES = [
    {"name": "Ilya Sutskever", "type": "person"},
    {"name": "Dario Amodei", "type": "person"},
    {"name": "Claude", "type": "technology"},
    {"name": "GraphSAGE", "type": "technology"},
    {"name": "AI安全性", "type": "concept"},
]

NEW_RELATIONS = [
    {"source": "Ilya Sutskever", "target": "OpenAI", "relation": "co-founded"},
    {"source": "Ilya Sutskever", "target": "GPT", "relation": "designed"},
    {"source": "Ilya Sutskever", "target": "Transformer", "relation": "co-invented"},
    {"source": "Dario Amodei", "target": "Anthropic", "relation": "founded"},
    {"source": "Dario Amodei", "target": "AI安全性", "relation": "researches"},
    {"source": "Dario Amodei", "target": "OpenAI", "relation": "formerly_at"},
    {"source": "Claude", "target": "Anthropic", "relation": "developed_by"},
    {"source": "Claude", "target": "自然言語処理", "relation": "applies"},
    {"source": "GraphSAGE", "target": "GNN", "relation": "subtype_of"},
    {"source": "GraphSAGE", "target": "深層学習", "relation": "subfield_of"},
    {"source": "AI安全性", "target": "Anthropic", "relation": "focus_of"},
]


def build_base_graph():
    """BASEグラフ（20エンティティ）を返す"""
    return BASE_ENTITIES, BASE_RELATIONS


def build_full_graph():
    """BASE + NEW の全グラフ（25エンティティ）を返す"""
    return BASE_ENTITIES + NEW_ENTITIES, BASE_RELATIONS + NEW_RELATIONS
