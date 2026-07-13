"""
RAG 回答の精度指標モジュール

ルールベース指標（常時算出・追加依存なし）:
  - faithfulness      忠実性: 回答の各文が検索コンテキストに根拠を持つ割合
  - answer_relevancy  回答関連性: 回答が質問の語彙をどれだけカバーしているか
  - context_relevancy コンテキスト関連性: 検索結果のうち質問に関係する割合
  - grounded_numbers  数値根拠率: 回答中の数値がコンテキストに存在する割合
  - overall           上記の加重平均（品質ゲートに使用）

LLM 判定（プロバイダが LLM の場合のみ併記）:
  - llm_faithfulness / llm_relevancy / llm_completeness / llm_overall
"""

import json
import re

# overall の重み
_WEIGHTS = {
    "faithfulness": 0.45,
    "answer_relevancy": 0.25,
    "context_relevancy": 0.15,
    "grounded_numbers": 0.15,
}

_STOPWORDS = {
    "こと", "もの", "ため", "よう", "これ", "それ", "あれ", "どこ", "だれ",
    "です", "ます", "ください", "について", "とは", "何", "教え", "して",
    "する", "いる", "ある", "なる", "また", "および", "など", "場合",
}


# ---------- 語彙抽出 ----------

def _terms(text: str) -> set[str]:
    """日本語対応の内容語抽出（漢字連続・カタカナ語・英数語 + 漢字バイグラム）"""
    if not text:
        return set()
    terms: set[str] = set()
    for m in re.findall(r"[一-龥々]+", text):
        if len(m) >= 2 and m not in _STOPWORDS:
            terms.add(m)
            terms.update(m[i:i + 2] for i in range(len(m) - 1))
        elif len(m) == 1:
            terms.add(m)
    for m in re.findall(r"[ァ-ヶー]{2,}", text):
        if m not in _STOPWORDS:
            terms.add(m)
    for m in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]+", text):
        terms.add(m.lower())
    return {t for t in terms if t not in _STOPWORDS}


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[。！？!?])\s*|\n", text or "") if s.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d[\d,\.]*", text or ""))


# ---------- ルールベース指標 ----------

def faithfulness(answer: str, context: str) -> float:
    """回答の各文について、内容語がコンテキストに含まれる割合を平均"""
    ctx_terms = _terms(context)
    if not ctx_terms:
        return 0.0
    sentence_scores = []
    for sent in _sentences(answer):
        s_terms = _terms(sent)
        if not s_terms:
            continue
        hit = sum(1 for t in s_terms if t in ctx_terms)
        sentence_scores.append(hit / len(s_terms))
    return round(sum(sentence_scores) / len(sentence_scores), 3) if sentence_scores else 0.0


def answer_relevancy(question: str, answer: str) -> float:
    """質問の内容語のうち回答（または回答文脈）に現れる割合"""
    q_terms = _terms(question)
    if not q_terms or not (answer or "").strip():
        return 0.0
    a_terms = _terms(answer)
    hit = sum(1 for t in q_terms if t in a_terms)
    return round(hit / len(q_terms), 3)


def context_relevancy(question: str, search_results: list[dict], context: str) -> float:
    """検索でヒットした項目のうち、質問語彙と重なるものの割合"""
    q_terms = _terms(question)
    if not q_terms:
        return 0.0
    items = []
    for r in search_results or []:
        items.append(str(r.get("name", "")))
    # search_results が空でもコンテキスト行単位で評価
    if not items:
        items = [ln for ln in (context or "").splitlines() if ln.strip()]
    if not items:
        return 0.0
    relevant = sum(1 for it in items if _terms(it) & q_terms)
    return round(relevant / len(items), 3)


def grounded_numbers(answer: str, context: str) -> float:
    """回答中の数値がコンテキストに存在する割合（数値なしは1.0）"""
    nums = _numbers(answer)
    if not nums:
        return 1.0
    ctx = (context or "").replace(",", "")
    hit = sum(1 for n in nums if n.replace(",", "") in ctx)
    return round(hit / len(nums), 3)


def compute_metrics(
    question: str,
    answer: str,
    context: str,
    search_results: list[dict] | None = None,
) -> dict:
    """ルールベースの精度指標一式を返す"""
    m = {
        "faithfulness": faithfulness(answer, context),
        "answer_relevancy": answer_relevancy(question, answer),
        "context_relevancy": context_relevancy(question, search_results or [], context),
        "grounded_numbers": grounded_numbers(answer, context),
    }
    m["overall"] = round(sum(m[k] * w for k, w in _WEIGHTS.items()), 3)
    return m


# ---------- LLM 判定（併用） ----------

_JUDGE_PROMPT = (
    "あなたはRAGシステムの回答を評価する審査員です。\n"
    "以下の質問・検索コンテキスト・回答を読み、次の3軸を0.0〜1.0で採点してください。\n"
    "- faithfulness: 回答がコンテキストの事実のみに基づいているか\n"
    "- relevancy: 回答が質問に答えているか\n"
    "- completeness: コンテキスト内の必要情報を過不足なく使っているか\n\n"
    "JSONのみを出力してください。例: {{\"faithfulness\": 0.9, \"relevancy\": 0.8, \"completeness\": 0.7}}\n\n"
    "## 質問\n{question}\n\n## コンテキスト\n{context}\n\n## 回答\n{answer}\n"
)


def llm_judge(llm_invoke, question: str, answer: str, context: str) -> dict | None:
    """LLM による多軸採点。失敗時は None（ルールベース値のみ使う）"""
    prompt = _JUDGE_PROMPT.format(
        question=question, context=(context or "")[:4000], answer=answer
    )
    try:
        raw = llm_invoke(prompt)
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))
        out = {}
        for key in ("faithfulness", "relevancy", "completeness"):
            val = float(data.get(key, 0.0))
            out[f"llm_{key}"] = max(0.0, min(1.0, round(val, 3)))
        out["llm_overall"] = round(
            (out["llm_faithfulness"] + out["llm_relevancy"] + out["llm_completeness"]) / 3, 3
        )
        return out
    except Exception:
        return None


# ---------- 一括評価用（期待回答との比較） ----------

def expected_match(answer: str, expected: str) -> float:
    """期待回答との内容語 F1"""
    a, e = _terms(answer), _terms(expected)
    if not a or not e:
        return 0.0
    inter = len(a & e)
    if inter == 0:
        return 0.0
    precision, recall = inter / len(a), inter / len(e)
    return round(2 * precision * recall / (precision + recall), 3)
