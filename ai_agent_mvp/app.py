import os
import json
from pathlib import Path
import streamlit as st

# -----------------------------
# Helpers (dict / BaseModel 공통 처리)
# -----------------------------
def sget(obj, key, default=None):
    """obj가 dict든 Pydantic(BaseModel)이든 동일하게 값 가져오기"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key, default)
    return default

def as_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # fallback
    try:
        return dict(obj)
    except Exception:
        return {"_raw": str(obj)}

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def render_score_breakdown(sb: dict):
    if not isinstance(sb, dict) or not sb:
        st.caption("score_breakdown 없음")
        return
    cols = st.columns(4)
    cols[0].metric("sim", sb.get("sim", 0))
    cols[1].metric("action", sb.get("action", 0))
    cols[2].metric("ctx", sb.get("ctx", 0))
    cols[3].metric("total", sb.get("total", 0))

def render_messages(messages):
    if not messages:
        st.caption("messages가 비어있습니다.")
        return

    # type 기준 정렬/탭 구성
    types = []
    for m in messages:
        t = (m.get("type") if isinstance(m, dict) else "") or "MESSAGE"
        if t not in types:
            types.append(t)

    tabs = st.tabs(types)
    for i, t in enumerate(types):
        with tabs[i]:
            for m in messages:
                if not isinstance(m, dict):
                    continue
                if (m.get("type") or "MESSAGE") != t:
                    continue

                title = m.get("title", "").strip()
                body = m.get("body", "").strip()

                st.markdown(
                    f"""
                    <div style="padding:14px;border:1px solid #e6e6e6;border-radius:12px;margin-bottom:12px;">
                      <div style="font-weight:700;font-size:16px;margin-bottom:8px;">{title}</div>
                      <div style="white-space:pre-wrap;line-height:1.6;">{body}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

def pick_customer(data_dir: Path):
    # 파일명이 customers.json / customer.json 둘 다 대응
    c1 = data_dir / "customers.json"
    c2 = data_dir / "customer.json"

    if c1.exists():
        customers = load_json(c1)
        if isinstance(customers, list) and customers:
            return customers, "customers.json"
    if c2.exists():
        customers = load_json(c2)
        # customer.json이 단일 객체면 list로 감싸기
        if isinstance(customers, dict):
            return [customers], "customer.json"
        if isinstance(customers, list) and customers:
            return customers, "customer.json"

    return [], None


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="CRM AI Agent", layout="wide")

    ROOT = Path(__file__).resolve().parent
    DATA_DIR = ROOT / "data"

    st.title("CRM AI Agent")

    # 사이드바 설정
    st.sidebar.header("설정")

    api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.sidebar.caption("API 키가 없어도 (룰/템플릿만 쓰는 노드라면) 실행은 됩니다. LLM 노드가 있으면 키가 필요합니다.")

    customers, loaded_file = pick_customer(DATA_DIR)
    if loaded_file:
        st.sidebar.caption(f"Loaded: data/{loaded_file}")

    # 고객 선택/입력
    st.subheader("1) 타겟 고객 선택")

    mode = st.radio("입력 방식", ["샘플 고객(JSON)", "수동 입력"], horizontal=True)

    if mode == "샘플 고객(JSON)":
        if not customers:
            st.error("data/customer(s).json 파일을 찾지 못했거나 비어있습니다.")
            st.stop()

        options = []
        for idx, c in enumerate(customers):
            cid = c.get("id", f"C{idx+1:03d}") if isinstance(c, dict) else f"C{idx+1:03d}"
            profile = (c.get("profile", {}) if isinstance(c, dict) else {}) or {}
            label = f"{cid} | {profile.get('age','?')}세 | {profile.get('skin_type','?')}"
            options.append((label, idx))

        label, idx = st.selectbox("고객 선택", options, format_func=lambda x: x[0])
        customer = customers[idx]

        profile = customer.get("profile", {}) or {}
        logs = customer.get("logs", {}) or {}
        context = customer.get("context", {}) or {}

        with st.expander("선택한 고객 데이터 보기", expanded=False):
            st.json(customer)

    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("나이", min_value=10, max_value=80, value=23, step=1)
            skin_type = st.selectbox("피부 타입", ["건성", "지성", "복합성", "민감성"], index=0)
        with col2:
            concerns = st.multiselect("피부 고민(복수 선택)", ["수분", "건조", "속당김", "트러블", "진정", "탄력", "주름", "미백", "잡티", "모공", "피지", "각질"])
        with col3:
            weather = st.selectbox("날씨 컨텍스트", ["건조", "습함", "추움", "더움", "보통"], index=0)
            season = st.selectbox("시즌", ["환절기", "봄", "여름", "가을", "겨울"], index=0)

        recent_search = st.text_area("최근 검색어(줄바꿈으로 여러 개)", "환절기 건조함 해결\n수분크림 추천")
        recent_viewed = st.text_area("최근 본 상품(줄바꿈으로 여러 개)", "워터뱅크 수분 크림")
        cart = st.text_area("장바구니 상품(줄바꿈으로 여러 개)", "")

        profile = {"age": age, "skin_type": skin_type, "concerns": concerns}
        logs = {
            "recent_search": [x.strip() for x in recent_search.splitlines() if x.strip()],
            "recent_viewed": [x.strip() for x in recent_viewed.splitlines() if x.strip()],
            "cart": [x.strip() for x in cart.splitlines() if x.strip()],
        }
        context = {"weather": weather, "season": season}

    st.divider()

    # 실행 버튼
    run = st.button("에이전트 실행", type="primary")

    if run:
        # LLM 노드가 있는 경우 키 없으면 안내
        # (노드가 룰 기반이면 없어도 실행됨)
        if not os.environ.get("GOOGLE_API_KEY"):
            st.warning("GOOGLE_API_KEY가 설정되지 않았습니다. LLM 호출 노드가 있으면 실패할 수 있습니다.")

        # import는 여기서 (환경변수 설정 이후에)
        from agent.state import AgentState
        from agent.graph import build_graph

        state = AgentState(
            user_profile=profile,
            user_logs=logs,
            context=context,
            max_retries=1,
        )

        app = build_graph()

        with st.spinner("실행 중..."):
            out = app.invoke(state)

        # out이 dict든 BaseModel이든 공통으로 처리
        persona = sget(out, "persona", "")
        persona_reason = sget(out, "persona_reason", "")
        selected_product = sget(out, "selected_product", {}) or {}
        candidate_products = sget(out, "candidate_products", []) or []
        generated_messages = sget(out, "generated_messages", []) or []
        is_valid = sget(out, "is_valid", False)
        retry_count = sget(out, "retry_count", 0)
        feedback = sget(out, "feedback", "")

        st.subheader("2) 최종 결과")

        left, right = st.columns([1, 1])

        with left:
            st.markdown("### 페르소나")
            st.write(persona)
            if persona_reason:
                st.caption(persona_reason)

            st.markdown("### 추천 상품")
            st.write(selected_product.get("name", ""))
            st.caption(f"product_id: {selected_product.get('product_id', '')} / brand_id: {selected_product.get('brand_id', '')}")

            if selected_product.get("decision_reason"):
                st.info(selected_product.get("decision_reason"))

            sb = selected_product.get("score_breakdown", {}) if isinstance(selected_product, dict) else {}
            render_score_breakdown(sb)

            st.markdown("### 검수 상태")
            st.write(f"is_valid: {is_valid}")
            st.write(f"retry_count: {retry_count}")
            if feedback:
                st.warning(feedback)

        with right:
            st.markdown("### 후보 Top-K")
            if candidate_products:
                table = [
                    {
                        "name": p.get("name", ""),
                        "similarity": p.get("similarity", 0),
                        "brand_id": p.get("brand_id", ""),
                        "product_id": p.get("product_id", ""),
                    }
                    for p in candidate_products
                    if isinstance(p, dict)
                ]
                st.dataframe(table, use_container_width=True, hide_index=True)
            else:
                st.caption("candidate_products가 비어있습니다.")

        st.markdown("### 메시지 3종")
        render_messages(generated_messages)

        with st.expander("디버그: 최종 상태 원문", expanded=False):
            st.json(as_dict(out))


if __name__ == "__main__":
    main()