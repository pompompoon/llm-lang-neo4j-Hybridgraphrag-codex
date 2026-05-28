"""
LLM プロバイダの一元管理（mock / ollama / gemini）
.env の PROVIDER を変えるだけで切り替わる
"""

import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "mock")
_ENV_KEYS = set(os.environ.keys())

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:4b")
OLLAMA_NO_THINK = os.getenv("OLLAMA_NO_THINK", "auto").lower()
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
if NEO4J_URI.startswith("neo4j://127.0.0.1") or NEO4J_URI.startswith("neo4j://localhost"):
    NEO4J_URI = NEO4J_URI.replace("neo4j://", "bolt://", 1)
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_CONFIGURED = "NEO4J_PASSWORD" in _ENV_KEYS


def get_chat_llm(temperature: float = 0.3):
    """チャット用 LLM を返す"""
    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_CHAT_MODEL, google_api_key=GEMINI_API_KEY,
            temperature=temperature)
    elif PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        reasoning = False if is_qwen3_family(OLLAMA_CHAT_MODEL) else None
        return ChatOllama(
            model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            reasoning=reasoning,
            num_ctx=OLLAMA_NUM_CTX,
            num_predict=OLLAMA_NUM_PREDICT,
            keep_alive="10m",
        )
    else:
        return MockLLM()


def get_provider_info() -> dict:
    return {"provider": PROVIDER, "chat_model": _model_name()}


def _model_name():
    if PROVIDER == "gemini":
        return GEMINI_CHAT_MODEL
    elif PROVIDER == "ollama":
        return OLLAMA_CHAT_MODEL
    return "mock"


def is_qwen3_family(model_name: str) -> bool:
    """Qwen3 / Qwen3.5 系はRAG用途ではthinkingを抑える。"""
    name = (model_name or "").lower()
    return name.startswith("qwen3") or name.startswith("qwen3.5")


class MockLLM:
    """スタブ LLM: LLM無しで LangGraph フロー全体を確認できる"""

    def invoke(self, messages, **kwargs):
        if isinstance(messages, list):
            text = str(messages[-1]) if messages else ""
        else:
            text = str(messages)

        # プロンプト種別を判定してそれらしい応答を返す
        if "キーワード" in text or "変換" in text:
            return _MockResponse("GraphSAGE GNN 構造埋め込み")
        elif "品質" in text or "評価" in text or "0.0" in text:
            return _MockResponse("0.85")
        else:
            return _MockResponse(
                "GraphSAGEはinductiveなGNNで、新規ノードへの埋め込み生成が可能です。"
                "近傍ノードの特徴を集約する関数を学習するため、学習時に存在しなかった"
                "ノードにも適用できます。これはRAGシステムでドキュメントが継続的に"
                "追加される場面で特に有効です。"
            )

    @property
    def _llm_type(self):
        return "mock"


class _MockResponse:
    def __init__(self, content: str):
        self.content = content

    def __str__(self):
        return self.content
