from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import AgentState

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_json(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))

def tokenize(text: str) -> List[str]:
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    return [t for t in text.lower().split() if t.strip()]

def keyword_similarity(query: str, doc: str) -> float:
    q = set(tokenize(query))
    d = set(tokenize(doc))
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)

# -----------------------
# Node 1: Persona (룰 기반)
# -----------------------
def node1_persona_classifier(state: AgentState) -> AgentState:
    profile = state.get("user_profile", {})
    logs = state.get("user_logs", {})

    skin = str(profile.get("skin_type", "")).lower()
    concerns = profile.get("concerns", [])
    if not isinstance(concerns, list):
        concerns = [str(concerns)]
    concern_text = " ".join([str(c) for c in concerns])

    recent_search = logs.get("recent_search", [])
    if not isinstance(recent_search, list):
        recent_search = [str(recent_search)]
    recent_search_text = " ".join([str(x) for x in recent_search])

    text = f"{skin} {concern_text} {recent_search_text}".lower()

    if any(k in text for k in ["민감", "자극", "트러블", "진정", "붉은기"]):
        persona = "P1_NATURAL_HEALING"
        reason = "민감/진정 니즈가 강하고 자극을 피하려는 탐색이 보입니다."
    elif any(k in text for k in ["탄력", "주름", "윤기", "안티에이징", "리프팅"]):
        persona = "P2_LUXURY_CARE"
        reason = "탄력/윤기 중심의 고기능 케어 니즈가 확인됩니다."
    elif any(k in text for k in ["수분", "건조", "보습", "속당김"]):
        persona = "P3_HYDRATION"
        reason = "건조/보습 탐색이 많아 수분 우선 페르소나로 분류됩니다."
    elif any(k in text for k in ["톤업", "잡티", "미백", "기미"]):
        persona = "P4_BRIGHTENING"
        reason = "톤/잡티 개선 니즈가 두드러집니다."
    elif any(k in text for k in ["각질", "피지", "모공", "블랙헤드"]):
        persona = "P5_CLEARING"
        reason = "피지/모공/각질 관련 관심도가 높습니다."
    else:
        persona = "P6_DAILY_BASIC"
        reason = "특정 고민보다 데일리 기본 케어 성향이 우세합니다."

    state["persona"] = persona
    state["persona_reason"] = reason
    return state

# -----------------------
# Node 2: Candidate Retrieval (하드필터 + 키워드 유사도)
# -----------------------
def node2_candidate_retrieval(state: AgentState) -> AgentState:
    products = load_json("products.json")
    persona = state.get("persona", "")

    profile = state.get("user_profile", {})
    logs = state.get("user_logs", {})
    ctx = state.get("context", {})

    concerns = profile.get("concerns", [])
    if not isinstance(concerns, list):
        concerns = [str(concerns)]

    q = " ".join([
        " ".join([str(c) for c in concerns]),
        " ".join(logs.get("recent_search", [])),
        " ".join(logs.get("recent_viewed", [])),
        str(ctx.get("weather", "")),
        str(ctx.get("season", "")),
    ])

    filtered = [p for p in products if persona in p.get("target_personas", [])]

    scored: List[Dict[str, Any]] = []
    for p in filtered:
        doc = " ".join([
            p.get("name", ""),
            p.get("summary", ""),
            " ".join(p.get("keywords", [])),
            " ".join(p.get("review_summary_bullets", [])),
        ])
        sim = keyword_similarity(q, doc)
        scored.append({**p, "similarity": sim})

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    state["candidate_products"] = scored[:5]
    return state

# -----------------------
# Node 3: Recommender (0.5/0.3/0.2)
# -----------------------
def node3_recommender(state: AgentState) -> AgentState:
    logs = state.get("user_logs", {})
    ctx = state.get("context", {})

    recent_viewed = set(logs.get("recent_viewed", []))
    cart = set(logs.get("cart", []))

    weather = str(ctx.get("weather", "")).lower()
    season = str(ctx.get("season", "")).lower()

    ranking = []
    for p in state.get("candidate_products", []):
        sim = float(p.get("similarity", 0.0)) * 0.50

        action_raw = 0.0
        if p.get("name") in recent_viewed:
            action_raw += 0.6
        if p.get("name") in cart:
            action_raw += 0.8
        action = action_raw * 0.30

        ctx_raw = 0.0
        tags = " ".join(p.get("season_tags", [])).lower()
        if season and season in tags:
            ctx_raw += 0.6
        if weather and weather in tags:
            ctx_raw += 0.4
        ctx_score = ctx_raw * 0.20

        total = sim + action + ctx_score
        breakdown = {
            "sim": round(sim, 4),
            "action": round(action, 4),
            "ctx": round(ctx_score, 4),
            "total": round(total, 4),
        }
        ranking.append((total, breakdown, p))

    ranking.sort(key=lambda x: x[0], reverse=True)
    best_total, best_breakdown, best = ranking[0]

    state["selected_product"] = {
        "product_id": best["product_id"],
        "brand_id": best["brand_id"],
        "name": best["name"],
        "score_breakdown": best_breakdown,
        "decision_reason": "니즈 적합도(유사도) + 최근 행동 + 시즌/날씨 적합도를 종합해 선정했습니다.",
        "raw": best,
    }
    return state

# -----------------------
# Node 4: Prompt Builder (팩트 + 브랜드룰 + 제약)
# -----------------------
def node4_prompt_builder(state: AgentState) -> AgentState:
    brand_rules = load_json("brand_rules.json")
    sp = state.get("selected_product", {})
    raw = sp.get("raw", {})
    brand_id = sp.get("brand_id")

    rule = brand_rules.get(brand_id, {})
    constraints = rule.get("constraints", {"title_max": 40, "body_max": 350})
    forbidden = rule.get("forbidden_words", [])
    required = rule.get("required_phrases", [])

    fact_data = {
        "product_name": raw.get("name", ""),
        "efficacy": raw.get("efficacy", []),
        "ingredients": raw.get("ingredients", []),
        "usage": raw.get("usage", ""),
        "review_summary": raw.get("review_summary_bullets", []),
    }
    state["fact_data"] = fact_data

    state["system_prompt"] = (
        "너는 아모레퍼시픽 마케팅 카피라이터다.\n"
        f"[브랜드 톤]\n{rule.get('tone_guide', '')}\n\n"
        "[제품 팩트(팩트만 사용)]\n"
        f"- 제품명: {fact_data['product_name']}\n"
        f"- 효능: {', '.join(fact_data['efficacy'])}\n"
        f"- 사용법: {fact_data['usage']}\n"
        f"- 리뷰 요약: {', '.join(fact_data['review_summary'])}\n\n"
        "[제약]\n"
        f"- 제목 {constraints.get('title_max', 40)}자 이내\n"
        f"- 본문 {constraints.get('body_max', 350)}자 이내\n"
        f"- 금기어: {', '.join(forbidden)}\n"
        f"- 필수 문구: {', '.join(required)}\n\n"
        f"[추천 근거]\n{sp.get('decision_reason', '')}\n"
    )
    return state

# -----------------------
# Node 5: Copywriter (LLM 1회 호출로 3전략 JSON 생성)
# -----------------------
def _get_llm() -> ChatGoogleGenerativeAI:
    # MVP는 flash로 고정 추천(쿼터/안정성)
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

def node5_copywriter_llm(state: AgentState) -> AgentState:
    llm = _get_llm()

    sp = state.get("selected_product", {})
    fact = state.get("fact_data", {})
    feedback = state.get("feedback", "")

    prompt = f"""
{state.get("system_prompt", "")}

[추가 지시]
- 아래 3가지 전략 메시지를 각각 1개씩 만들어라.
  1) TECH_SPEC: 효능/팩트 중심
  2) EMPATHY: 상황 공감/고민 해결 중심
  3) BENEFIT_TIP: 실용/루틴/사용 팁 중심
- 반드시 '제품 팩트' 범위에서만 말할 것(없는 효능/성분 만들지 말 것)
- 피드백이 있으면 반영: {feedback}

[출력은 JSON만]
형식:
{{
  "messages": [
    {{"type":"TECH_SPEC","title":"...","body":"..."}},
    {{"type":"EMPATHY","title":"...","body":"..."}},
    {{"type":"BENEFIT_TIP","title":"...","body":"..."}}
  ]
}}
"""
    res = llm.invoke(prompt).content.strip()

    # LLM이 ```json``` 같은 걸 붙일 수 있어 방어
    res = re.sub(r"^```json\s*|\s*```$", "", res.strip(), flags=re.IGNORECASE)

    try:
        obj = json.loads(res)
        msgs = obj.get("messages", [])
        if not isinstance(msgs, list):
            msgs = []
    except Exception:
        # 파싱 실패 시 최소 안전 fallback
        msgs = [
            {"type": "TECH_SPEC", "title": f"{fact.get('product_name','')} 핵심 케어", "body": f"{', '.join(fact.get('efficacy',[])[:2])} 중심. {fact.get('usage','')}"},
            {"type": "EMPATHY", "title": "요즘 피부 컨디션", "body": f"흔들릴 때 부담 없이 루틴에. {fact.get('usage','')}"},
            {"type": "BENEFIT_TIP", "title": "오늘 루틴 팁", "body": f"토너 다음 단계로 간단히. {fact.get('usage','')}"},
        ]

    state["generated_messages"] = msgs
    return state

# -----------------------
# Node 6: Compliance Guardian (규칙 검수 + feedback + retry)
# -----------------------
def _validate_messages(messages: List[Dict[str, Any]], rule: Dict[str, Any]) -> Tuple[bool, List[str]]:
    violations: List[str] = []
    c = rule.get("constraints", {})
    title_max = int(c.get("title_max", 40))
    body_max = int(c.get("body_max", 350))
    forbidden = rule.get("forbidden_words", [])
    required = rule.get("required_phrases", [])

    for i, m in enumerate(messages):
        title = str(m.get("title", ""))
        body = str(m.get("body", ""))
        if len(title) > title_max:
            violations.append(f"[{i}] 제목 길이 초과({len(title)}/{title_max})")
        if len(body) > body_max:
            violations.append(f"[{i}] 본문 길이 초과({len(body)}/{body_max})")
        for w in forbidden:
            if w and (w in title or w in body):
                violations.append(f"[{i}] 금기어 포함: {w}")
        for r in required:
            if r and (r not in body):
                violations.append(f"[{i}] 필수 문구 누락: {r}")

    return (len(violations) == 0), violations

def node6_compliance_guardian(state: AgentState) -> AgentState:
    brand_rules = load_json("brand_rules.json")
    brand_id = state.get("selected_product", {}).get("brand_id")
    rule = brand_rules.get(brand_id, {})

    ok, violations = _validate_messages(state.get("generated_messages", []), rule)

    if ok:
        state["is_valid"] = True
        state["feedback"] = "PASS"
        return state

    state["is_valid"] = False
    state["retry_count"] = int(state.get("retry_count", 0)) + 1
    state["feedback"] = " / ".join(violations) + " -> 제목/본문 줄이고, 금기어 제거, 필수 문구 추가"
    return state

# -----------------------
# Node 7: Final Output
# -----------------------
def node7_final_output(state: AgentState) -> AgentState:
    state["final_output"] = {
        "persona": state.get("persona"),
        "persona_reason": state.get("persona_reason"),
        "selected_product": {
            "product_id": state.get("selected_product", {}).get("product_id"),
            "brand_id": state.get("selected_product", {}).get("brand_id"),
            "name": state.get("selected_product", {}).get("name"),
            "score_breakdown": state.get("selected_product", {}).get("score_breakdown"),
            "decision_reason": state.get("selected_product", {}).get("decision_reason"),
        },
        "messages": state.get("generated_messages", []),
        "retry_count": state.get("retry_count", 0),
        "is_valid": state.get("is_valid", False),
        "feedback": state.get("feedback", ""),
    }
    return state