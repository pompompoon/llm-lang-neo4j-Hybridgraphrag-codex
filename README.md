# LLMGraphRAG

第2世代 GraphRAG（GraphRAG Local LLM）に **GraphSAGE による inductive な構造埋め込み** を組み込んだ拡張プロジェクト。

知識グラフのエンティティを「テキストの意味」だけでなく「グラフ構造上の位置」も反映したベクトルで表現し、ハイブリッド検索・リンク予測（グラフ補完）を実現する。

GraphSAGE に加えて GAT（Graph Attention Network）も選択できる。Web UI のセットアップ欄で `GraphSAGE` / `GAT` を切り替えて学習できる。
■WEB画面
<img width="1896" height="820" alt="image" src="https://github.com/user-attachments/assets/7f2d7843-a201-4700-876b-7ca8bfaf42a1" />
<img width="1912" height="819" alt="image" src="https://github.com/user-attachments/assets/a51f4114-a1ad-4e16-b197-c21a58e8f4f7" />
<img width="1913" height="834" alt="image" src="https://github.com/user-attachments/assets/8d0d141d-a41f-4251-8095-400a71f5316e" />
<img width="1903" height="828" alt="image" src="https://github.com/user-attachments/assets/6a293e26-c490-4cb3-9aea-fd467613cfcd" />
<img width="1910" height="827" alt="image" src="https://github.com/user-attachments/assets/d6218600-4df9-4dd1-ad6c-c4a92c490084" />




## GraphSAGE と GAT の比較

| 項目 | GraphSAGE | GAT |
|---|---|---|
| 近傍集約 | 平均などで近傍特徴を集約 | attentionで重要な近傍を重み付け |
| 速度 | 軽い・速い | やや重い |
| 小規模グラフでの安定性 | 安定しやすい | 効く場合もあるが過学習しやすい |
| ノイズの多い関係 | 平均化でならす | 重要関係を選べる可能性 |
| 説明性 | 低め | attention重みで多少解釈しやすい |
| 推奨用途 | 既定・安定運用 | 比較実験・関係の重要度を見たい場合 |

このプロジェクトでは `GraphSAGE` を既定値にし、`GAT` は比較実験用として追加している。

## なぜ GraphSAGE か

第2世代 GraphRAG の検索は、エンティティの**テキスト埋め込み**（nomic-embed-text 等）によるベクトル類似検索と、Cypher によるグラフ近傍展開の組み合わせだった。グラフ構造は「その場で辿る」だけで、**ベクトル空間には埋め込まれていなかった**。

GraphSAGE を加えることで、グラフ構造そのものを埋め込みに反映できる。さらに GraphSAGE は **inductive**（帰納的）であり、これが GraphRAG にとって決定的に重要となる。

### transductive と inductive の違い

| | DeepWalk / node2vec（transductive） | GraphSAGE（inductive） |
|---|---|---|
| 学習するもの | ノードID→ベクトルのルックアップ表 | 近傍を集約・変換する関数 |
| 新規ノードの埋め込み | 不可（グラフ全体の再学習が必要） | 可能（再学習不要） |
| GraphRAG との相性 | 悪い（文書追加のたび再学習） | 良い（増分更新に対応） |

GraphRAG はドキュメントが継続的に追加され、知識グラフに新エンティティが増え続ける。transductive 手法は新エンティティのたびに再学習が必要で現実的でない。GraphSAGE は学習済みの集約関数を新エンティティにも適用でき、**再学習なしに埋め込みを生成できる**。

## アーキテクチャ

```
知識グラフ（エンティティ + リレーション）
  ↓
graph_builder.py        ← PyTorch Geometric の Data に変換
  ↓
GraphSAGEEncoder        ← 2層 SAGEConv（近傍集約 mean）
  ↓ 自己教師あり学習（リンク予測）
train.py                ← 既存エッジ=正例 / 非エッジ=負例 で学習
  ↓
InductiveEngine         ← 学習済みモデルで inductive 推論
  ├─ encode_all                全エンティティの構造埋め込み生成
  ├─ find_similar_by_structure 構造ベース類似検索
  ├─ hybrid_search             テキスト + 構造のハイブリッド検索
  └─ suggest_missing_links     リンク予測によるグラフ補完
  ↓
langgraph_agent.py      ← LangGraph 自律ループ（4ノード）
  ├─ クエリ分析ノード         LLM で検索クエリを最適化
  ├─ GraphSAGE検索ノード      ハイブリッド検索で関連エンティティ取得
  ├─ 回答生成ノード           LLM で回答生成
  └─ 品質チェックノード        LLM 自己評価 → 品質NGなら再検索ループ
  ↓
config.py               ← LLM プロバイダの一元管理（.env で切替）
  └─ PROVIDER: mock / ollama / gemini
  ↓
main.py（FastAPI）      ← API として公開
```

## LangGraph チャット層

第2世代 GraphRAG と同じ LangGraph 4ノード構成。違いは検索ノードが **GraphSAGE の構造埋め込みハイブリッド検索**を使う点。

```
クエリ分析 ──> GraphSAGE検索 ──> 回答生成 ──> 品質チェック
                  ↑                              │
                  │   品質NG（スコア<0.7）        │
                  └──────────────────────────────┘
                       最大2回まで再検索（自律ループ）
```

### LLM プロバイダの切り替え

`config.py` が LLM プロバイダを一元管理する。`.env` の `PROVIDER` を変えるだけで切り替わる。

| PROVIDER | 用途 | 必要なもの |
|---|---|---|
| `mock` | 動作確認・デモ | なし（スタブ LLM） |
| `ollama` | ローカル LLM | Ollama 起動 + `langchain-ollama` |
| `gemini` | クラウド LLM | API キー + `langchain-google-genai` |

## ファイル構成

```
llm-graph-rag/
├── backend/
│   ├── graphsage_model.py     # GraphSAGE モデル（SAGEConv + リンク予測）
│   ├── graph_builder.py       # 知識グラフ → PyG Data 変換 / Neo4j 統合関数
│   ├── neo4j_loader.py        # 第2世代 GraphRAG の Neo4j 読み書き
│   ├── seed_neo4j.py          # 合成データを第2世代形式で Neo4j に投入
│   ├── sample_data.py         # 合成知識グラフ（AI業界、BASE + NEW）
│   ├── train.py               # リンク予測による自己教師あり学習
│   ├── inductive_engine.py    # inductive 推論エンジン
│   ├── config.py              # LLM プロバイダの一元管理（mock/ollama/gemini）
│   ├── langgraph_agent.py     # LangGraph 自律ループ（4ノード）
│   ├── evaluate_inductive.py  # inductive 性の実証スクリプト
│   ├── main.py                # FastAPI サーバー
│   ├── static/
│   │   └── index.html         # Web UI（単一HTML、4タブ構成）
│   ├── .env.example           # 環境設定サンプル
│   ├── .env                   # 環境設定（gitignore推奨）
│   └── requirements.txt
└── README.md
```

## セットアップ

### 前提条件

- Python 3.11（Anaconda推奨）
- PyTorch 2.1+
- PyTorch Geometric 2.4+
- Ollama（`PROVIDER=ollama` の場合）

### 1. インストール

```bash
cd llm-graph-rag/backend

# PyTorch（CPU版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric
pip install torch-geometric

# 残りのパッケージ
pip install -r requirements.txt
```

### 2. 環境設定

```bash
cp .env.example .env
```

デフォルトは `PROVIDER=mock`（LLM不要で動作確認可能）。

Ollama を使う場合：

```env
PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen2.5:7b,qwen3:4b
```

Gemini を使う場合：

```env
PROVIDER=gemini
GEMINI_API_KEY=取得したAPIキー
GEMINI_CHAT_MODEL=gemini-2.5-flash
```

### 3. Ollama モデル準備（PROVIDER=ollama の場合のみ）

```bash
ollama pull qwen2.5:7b
ollama pull qwen3:4b

```

## 使い方

### 1. inductive 性の実証（推奨：まずこれを実行）

```bash
cd backend
python evaluate_inductive.py
```

BASE グラフ（20エンティティ）で学習 → NEW エンティティ（5個）を追加して再学習なしに埋め込み生成、までを一気に実行する。

実行結果の例：

```
[Phase 1] BASE グラフ（20エンティティ）で学習
  学習時間: 0.8秒
  AUC: 1.0000

[Phase 2] NEW エンティティ（5個）を追加 → 再学習なしに推論
  推論時間: 0.002秒（再学習なし）

[Phase 3] 新規エンティティの類似検索
  Ilya Sutskever の構造的に類似するエンティティ:
    - Sam Altman (person): 0.9123
    - Demis Hassabis (person): 0.8456

[Phase 4] リンク予測
  Ilya Sutskever ↔ Sam Altman: 0.9534 (✓ 既存)
  Claude ↔ GPT: 0.7821 (✗ 未接続)
```

### 2. 学習単体

```bash
python train.py
```

### 3. API サーバー / Web UI

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

ブラウザで `http://localhost:8000/` を開く。`npm` 不要 — FastAPI が単一 HTML を配信する。

## Web UI の操作方法

### 初回セットアップ

1. 左サイドバーの「セットアップ」で学習データを選択（合成データ or Neo4j）
2. 「Train + Embed」ボタンをクリック
3. 学習完了後、統計情報が更新される

### ドキュメント追加

左サイドバーの「ドキュメント追加」から、任意のテキストを現在のグラフへ追加できる。

1. テキストエリアに追加したい本文を貼り付ける
2. 必要に応じてソース名を入力する
3. 「追加して埋め込み更新」をクリック
4. 文書はチャンク化され、抽出されたエンティティと `mentions` リレーションで接続される

追加後は GraphSAGE モデルを再学習せず、更新後のグラフに対して構造埋め込みを再生成する。

### チャットタブ

LangGraph 自律ループ付き RAG チャット。

1. テキストボックスに質問を入力して Enter
2. GraphSAGE ハイブリッド検索で関連エンティティを取得
3. LLM が検索結果をもとに回答を生成
4. 品質スコア（0.0〜1.0）が表示される
5. 品質NGの場合は自動でリトライ（最大2回）

### ハイブリッド検索タブ

テキスト埋め込みと構造埋め込みの重みを調整して検索。

1. 検索クエリを入力
2. スライダーで `text_weight` を調整
   - 左寄り（0.0）: 構造埋め込みのみ（グラフ上の位置が近いエンティティ）
   - 右寄り（1.0）: テキスト埋め込みのみ（名前が似ているエンティティ）
   - 中央（0.5）: 両者を均等に統合
3. 「検索」をクリック
4. 各結果に total / text / struct のスコアが表示される

### リンク予測タブ

2エンティティ間のリンク確率予測とグラフ補完の提案。

**リンク予測：**

1. エンティティA、Bを入力
2. 「予測」をクリック
3. リンク確率（%）と既存エッジの有無が表示される

**グラフ補完：**

1. エンティティ名を入力
2. 「提案を取得」をクリック
3. まだ接続されていないが接続されるべきエンティティが確率順に表示される

### グラフタブ

Canvas 力学シミュレーションで知識グラフを可視化。

- ノードの色はエンティティタイプごと（person=赤, organization=ティール, technology=紫, concept=青, event=ピンク）
- 新規エンティティ（NEW）は緑枠で表示
- エッジにリレーション名が表示される
- 「更新」ボタンで最新データを再取得

## 第2世代 GraphRAG（Neo4j）との接続

### データ供給元の2系統

| 供給元 | 用途 |
|---|---|
| 合成データ（`sample`） | inductive 性の実証、デモ |
| Neo4j（`neo4j`） | 第2世代の実データで学習・推論 |

### Neo4j データで学習する

Swagger UI（`http://localhost:8000/docs`）から：

```
POST /api/graphsage/train
Body: {"data_source": "neo4j", "neo4j_uri": "bolt://localhost:7687", "neo4j_password": "password"}
```

### 構造埋め込みを Neo4j に書き戻す

GraphSAGE が生成した構造埋め込みを、第2世代の Entity ノードに `struct_embedding` プロパティとして書き戻す。

```
POST /api/neo4j/writeback
Body: {"neo4j_uri": "bolt://localhost:7687", "property_name": "struct_embedding"}
```

書き戻し後、Neo4j Browser で確認：

```cypher
MATCH (e:Entity) RETURN e.name, e.struct_embedding LIMIT 5
```

### 第2世代 GraphRAG なしで Neo4j 連携を動作実証する

`seed_neo4j.py` で合成知識グラフを Neo4j に投入できる。

```bash
python seed_neo4j.py --uri neo4j://127.0.0.1:7687 --password yourpassword

# 既存データをクリアしてから投入する場合
python seed_neo4j.py --uri neo4j://127.0.0.1:7687 --password yourpassword --wipe
```

投入後、Swagger UI で以下を順に実行：

1. `POST /api/graphsage/train`（data_source: "neo4j"）
2. `POST /api/graphsage/embed`
3. `POST /api/neo4j/writeback`

### データの流れ（Neo4j 統合時）

```
第2世代 GraphRAG の Neo4j
  (:Entity {name, type, embedding})-[:RELATES]->(:Entity)
        ↓ ① neo4j_loader.py が Cypher で読み出し
        ↓    Entity.embedding をノード特徴量に使う
  graph_builder.build_pyg_data()
        ↓ ② PyG Data に変換
  GraphSAGE 学習・推論
        ↓ ③ 構造埋め込みを生成
  neo4j_loader.write_structure_embeddings()
        ↓ ④ Entity.struct_embedding として書き戻し
第2世代 GraphRAG の Neo4j（構造埋め込みが追加された状態）
```

## .env 設定リファレンス

```env
# LLM プロバイダ: mock / ollama / gemini
PROVIDER=mock

# Ollama (PROVIDER=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen2.5:7b

# Gemini (PROVIDER=gemini)
GEMINI_API_KEY=
GEMINI_CHAT_MODEL=gemini-2.5-flash
```

## API リファレンス

| メソッド | エンドポイント | 説明 |
|---|---|---|
| GET | `/` | Web UI（単一 HTML） |
| GET | `/api/health` | ヘルスチェック（LLM プロバイダも表示） |
| POST | `/api/graphsage/train` | BASE グラフで学習 |
| POST | `/api/graphsage/embed` | 全エンティティの構造埋め込み生成（inductive） |
| POST | `/api/chat` | LangGraph 自律ループ付き RAG チャット |
| POST | `/api/documents/add` | テキスト文書をチャンク・エンティティとしてグラフに追加 |
| GET | `/api/entities/{name}/similar` | 構造ベース類似エンティティ検索 |
| POST | `/api/search/hybrid` | ハイブリッド検索（テキスト + 構造） |
| POST | `/api/links/predict` | 2エンティティ間のリンク予測 |
| GET | `/api/links/suggest/{name}` | グラフ補完の提案 |
| GET | `/api/graph/stats` | グラフ統計 |
| GET | `/api/graph/data` | グラフ可視化データ |
| POST | `/api/neo4j/writeback` | 構造埋め込みを Neo4j に書き戻し |

## ハイブリッド検索の設計

`text_weight` パラメータでテキスト埋め込みと構造埋め込みの寄与を調整できる：

- `text_weight=1.0`: テキスト埋め込みのみ（意味的に似たエンティティ）
- `text_weight=0.0`: 構造埋め込みのみ（グラフ構造的に似たエンティティ）
- `text_weight=0.5`: 両者を均等に統合

「意味は違うがグラフ上で強く結びつくエンティティ」を発見できるのが、構造埋め込みを加える価値。

## 実証結果

`evaluate_inductive.py` の実行例：

- **学習時間**（BASE 20エンティティ）: 約 1 秒
- **推論時間**（NEW 5個を含む 25エンティティ）: 約 0.002 秒
- リンク予測の検証 AUC: 1.0（合成データのため高い）
- 新規エンティティ「Ilya Sutskever」の構造埋め込みが「Sam Altman」と高類似度 → どちらも OpenAI に強く結びつく人物で、グラフ構造を正しく捉えている

新規エンティティの追加に**再学習が不要**であることが、ドキュメントが増え続ける GraphRAG に GraphSAGE が適する理由。

## トラブルシューティング

**PyTorch Geometric インストールエラー** → PyTorch を先にインストールしてから `pip install torch-geometric` を実行。CUDA版が不要なら `--index-url https://download.pytorch.org/whl/cpu` を付ける。

**`PROVIDER=mock` でチャットの回答が固定文** → 正常動作。mock はスタブLLMなので固定文を返す。実際のLLMを使うには `PROVIDER=ollama` または `PROVIDER=gemini` に変更。

**Neo4j 接続エラー** → Neo4j が起動しているか確認。`neo4j` パッケージがインストールされているか確認（`pip install neo4j`）。

**学習が遅い** → `epochs` パラメータを下げる（デフォルト200）。合成データ（20ノード）なら100でも十分。

**Web UI が表示されない** → `backend/static/index.html` が存在するか確認。`uvicorn` の起動ディレクトリが `backend/` であることを確認。

## 制限事項

- 合成データ（AI業界の知識グラフ）を使用。実データでの検証は別途必要
- テキストマッチは簡易実装（エンティティ名の部分一致）。実システムでは nomic-embed-text 等の実埋め込みに置き換える
- モデルはメモリ保持。本番では学習済みモデルの永続化（`torch.save` / `torch.load`）が必要
- リンク予測は無向グラフ前提。関係の種類（develops, led_by 等）は区別していない

## ライセンス

MIT License
