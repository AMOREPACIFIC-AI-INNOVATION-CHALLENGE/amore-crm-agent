from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

class AgentState(TypedDict):
    user_info: Dict
    persona: str
    product: Dict
    draft_msg: str
    final_msg: str
    feedback: str
    retry_count: int

MOCK_PRODUCTS = [
    {"id": 1, "name": "레티놀 시카 흔적 앰플", "desc": "트러블 흔적 케어, 저자극 효능", "target": "트러블"},
    {"id": 2, "name": "그린티 씨드 히알루론산 세럼", "desc": "속건조 해결, 수분 급속 충전", "target": "건조"},
    {"id": 3, "name": "블랙티 유스 인핸싱 앰플", "desc": "피로 회복, 탄력 개선, 안티에이징", "target": "탄력"},
]

def analyze_persona_node(state: AgentState):
    user = state["user_info"]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    prompt = f"""
이 고객의 데이터를 보고 가장 적절한 '마케팅 페르소나'를 한 단어로 정의해.
고객 데이터: {user}
출력 형식: [페르소나명]
예시: [합리적 꼼꼼이], [트렌드 팔로워]
"""
    res = llm.invoke(prompt).content.strip()
    if res.startswith("[") and res.endswith("]"):
        res = res[1:-1].strip()
    return {"persona": res}

def recommend_product_node(state: AgentState):
    user_worry = state["user_info"].get("worry", "")
    selected = MOCK_PRODUCTS[0]
    for p in MOCK_PRODUCTS:
        if p["target"] in user_worry:
            selected = p
            break
    return {"product": selected}

def generate_message_node(state: AgentState):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    product = state["product"]
    persona = state["persona"]
    feedback = state.get("feedback", "")

    prompt = f"""
당신은 아모레퍼시픽 마케터입니다. 아래 정보를 바탕으로 고객에게 보낼 카톡 메시지를 작성하세요.

[타겟 고객]: {persona}
[추천 제품]: {product['name']} ({product['desc']})
[이전 피드백(있다면 수정 반영)]: {feedback}

[조건]
1. 친근하고 자연스러운 말투 사용.
2. 공백 포함 100자 이내로 짧게.
3. 이모지 1~2개 사용.

메시지 내용만 딱 출력하세요.
"""
    res = llm.invoke(prompt).content.strip()
    return {"draft_msg": res}

def validate_message_node(state: AgentState):
    msg = state["draft_msg"]

    if len(msg) > 100:
        return {
            "feedback": "메시지가 너무 깁니다. 100자 이내로 줄이세요.",
            "final_msg": None,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    forbidden = ["최고", "무조건", "100%"]
    for word in forbidden:
        if word in msg:
            return {
                "feedback": f"'{word}'는 과장 광고 위험이 있어 사용할 수 없습니다.",
                "final_msg": None,
                "retry_count": state.get("retry_count", 0) + 1,
            }

    return {"feedback": "PASS", "final_msg": msg}

def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("analyzer", analyze_persona_node)
    workflow.add_node("recommender", recommend_product_node)
    workflow.add_node("generator", generate_message_node)
    workflow.add_node("validator", validate_message_node)

    workflow.set_entry_point("analyzer")
    workflow.add_edge("analyzer", "recommender")
    workflow.add_edge("recommender", "generator")
    workflow.add_edge("generator", "validator")

    def check_result(state: AgentState):
        if state["feedback"] == "PASS":
            return "end"
        if state.get("retry_count", 0) >= 3:
            return "end"
        return "retry"

    workflow.add_conditional_edges(
        "validator",
        check_result,
        {"end": END, "retry": "generator"},
    )

    return workflow.compile()