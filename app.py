import streamlit as st
import os

# 페이지 설정
st.set_page_config(page_title="CRM AI Agent", layout="wide")
st.title("⚡ [Agent 10] 초개인화 메시지 생성기")

# 1. API 키 입력
api_key = st.sidebar.text_input("Google API Key", type="password")

# 2. 고객 정보 입력
st.subheader("1. 타겟 고객 선택")
col1, col2 = st.columns(2)
with col1:
    user_name = st.text_input("이름", "김민지")
    age = st.number_input("나이", 25, 50, 29)
with col2:
    skin_type = st.selectbox("피부 타입", ["건성", "지성", "복합성"])
    worry = st.selectbox("피부 고민", ["건조", "트러블", "탄력", "미백"])

# 3. 실행 버튼
if st.button("🚀 AI 에이전트 실행", type="primary"):
    if not api_key:
        st.error("API 키를 먼저 입력해주세요!")
    else:
        # 키 먼저 주입 (중요)
        os.environ["GOOGLE_API_KEY"] = api_key

        # build_agent는 여기서 import (키 적용 타이밍 보장)
        from agent import build_agent

        app = build_agent()

        inputs = {
            "user_info": {
                "name": user_name,
                "age": age,
                "skin_type": skin_type,
                "worry": worry
            },
            "retry_count": 0
        }

        with st.status("AI 에이전트가 작업 중입니다...", expanded=True) as status:
            final_output = None
            last_state = None  # 마지막 state 저장(재시도 횟수 확인용)

            for event in app.stream(inputs):
                for node_name, result in event.items():
                    last_state = result

                    if node_name == "analyzer":
                        st.write(f"🕵️ 페르소나 분석 완료: {result['persona']}")
                    elif node_name == "recommender":
                        prod = result["product"]
                        st.write(f"🎁 상품 선정 완료: {prod['name']} (사유: {prod['target']} 케어)")
                    elif node_name == "generator":
                        st.markdown(f"✍️ 메시지 생성 시도: `{result['draft_msg']}`")
                    elif node_name == "validator":
                        if result["feedback"] == "PASS":
                            st.success("✅ 검수 통과!")
                            final_output = result["final_msg"]
                        else:
                            st.warning(f"❌ 검수 반려: {result['feedback']} -> 재생성 중...")

            status.update(label="작업 완료!", state="complete", expanded=False)

        if final_output:
            st.divider()
            st.subheader("💌 최종 CRM 메시지")
            st.info(final_output)
            st.caption("이 메시지는 브랜드 톤앤매너 검수를 통과했습니다.")
        else:
            # retry_count는 last_state에서 확인해야 정확함
            retries = 0
            if isinstance(last_state, dict):
                retries = int(last_state.get("retry_count", 0))

            if retries >= 3:
                st.error("최대 수정 횟수를 초과하여 생성에 실패했습니다.")
            else:
                st.error("메시지 생성이 완료되지 않았습니다. (상태를 확인해주세요)")