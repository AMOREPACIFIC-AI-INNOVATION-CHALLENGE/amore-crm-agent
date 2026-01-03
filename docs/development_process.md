# **AI Agent 개발 구현 과정**    
<br>

# 1. 개요 및 기술 스택

본 프로젝트는 고객 데이터를 분석하여 페르소나를 도출하고 최적의 상품 추천과 함께 브랜드 톤앤매너를 준수하는 마케팅 메시지를 자동 생성하는 **AI 큐레이터** 시스템이다. Python 기반의 **LangGraph**를 도입하여 단순 생성이 아닌 '추론 → 생성 → 검수 → 수정'의 순환적 흐름을 가진 지능형 에이전트를 구축한다.

### 1) 기술 스택

확정된 기술 스택은 다음과 같다.

- **Orchestration:** **LangGraph** (State 관리, 순환형 그래프 제어, 멀티 에이전트 조율)
- **LLM Model:**
    - **Logic/Generation:** **Google Gemini 3.0 Pro** (복합 추론, 페르소나 분석, 고품질 메시지 생성)
    - **Utility/Validation:** **Google Gemini 3.0 Flash** (빠른 데이터 분류, 반복 가드레일 검수)
- **Database:**
    - **Vector DB (ChromaDB):** 제품 상세 정보, 리뷰, 고객 니즈 임베딩 (Semantic Search)
    - **RDB (PostgreSQL):** 브랜드 톤앤매너 규칙(금기어/권장표현), 고객 로그, 피드백 데이터
- **Application Layer:**
    - **Backend:** **FastAPI** (Python REST API)
    - **Frontend:** **Next.js** (Production), **Streamlit** (Prototype)

<br>

# 2. 시스템 아키텍처


전체 시스템은 [Data Layer] - [AI Logic Layer] - [Application Layer]의 3계층 구조로 설계되었으며 각 단계는 모듈화되어 독립성과 확장성을 가진다.

### 2.1 데이터 및 AI 모델 연계 방식

LLM의 환각을 억제하고 데이터 무결성을 확보하기 위해 RAG와 하이브리드 검색을 핵심으로 적용한다.

1. **고객 데이터 로드 및 정규화**
    - 고객 프로필(피부 타입/고민/연령대 등)과 행동 로그(최근 검색/조회/장바구니/구매 이력)를 RDB에서 조회한 뒤, 추천/분석에 필요한 형태로 정규화하여 에이전트 전역 상태로 주입한다.
2. **하이브리드 검색**
    - **Semantic Search (Vector DB):** "환절기 건조함 해결"과 같은 추상적 니즈를 임베딩하여 유사한 제품을 검색한다.
    - **Syntactic Search (Metadata Filtering):** 페르소나, 카테고리 등 명확한 속성은 **Hard Filter**로 적용하여 검색 범위를 먼저 제한한다.
3. **규칙 기반 통제**
    - 브랜드 고유의 톤앤매너, 금기어, 필수 표기 사항은 RDB에서 정형 데이터로 로드하여 시스템 프롬프트에 강제로 주입한다.

<br>

# 3. LangGraph 기반 에이전트 로직 설계


LangGraph의 StateGraph를 활용하여 에이전트의 워크플로우를 정의한다. 각 노드는 전역 상태를 공유하며, 검수 실패 시 이전 단계로 되돌아가는 **Self-Correction Loop**가 핵심이다.

### 3.1 전역 상태 정의

모든 노드는 아래의 JSON 스키마를 공유하며 데이터를 업데이트한다.

```python
class AgentState(TypedDict):
    user_profile: Dict       # 고객 프로필
    user_logs: Dict          # 고객 로그
    context: Dict            # 날씨, 시즌, 기념일
    persona: str             # 분석된 페르소나 ID
    persona_reason: str      # 페르소나 선정 근거 (CoT)
    candidate_products: List # Vector Search 결과 (Top-K)
    selected_product: Dict   # 최종 선정 제품 및 점수 Breakdown
    generated_messages: List # 생성된 메시지 후보 3종
    feedback: str            # 검수 결과에 따른 수정 지시사항
    retry_count: int         # 재생성 시도 횟수 (Max 5)
```

### 3.2 노드별 상세 알고리즘

### **Node 1. Persona Classifier (페르소나 분석)**

- **기능:** 고객의 프로필과 행동 로그를 입력받아 6가지 페르소나 중 하나로 분류하고 논리적 근거를 생성한다.
- **Logic:**
    - `Input:` User Profile + User Logs
    - `Process:` Gemini 3.0 Pro를 활용한 CoT(Chain-of-Thought) 추론
    - `Output:` Persona Tag + Reasoning Text

```verilog
ALGORITHM Node1_Persona_Classifier
    INPUT: User_Profile, User_Logs
    OUTPUT: Persona_Tag, Reasoning_Text

    // 1. 입력 데이터 통합 (Logic: User Profile + User Logs)
    Integrated_Data ← Merge(User_Profile, User_Logs)

    // 2. Gemini 3.0 Pro 기반 CoT 추론 수행
    // Process: Gemini 3.0 Pro, CoT(Chain-of-Thought)
    Raw_Output ← LLM_Inference(
        Model="Gemini-3.0-Pro",
        Method="Chain-of-Thought",
        Task="Classify user into one of 6 personas and provide reasoning",
        Input=Integrated_Data
    )

    // 3. 결과 파싱 (Output: Persona Tag + Reasoning Text)
    Persona_Tag    ← Extract_Tag(Raw_Output)       // 6가지 중 1개 선택
    Reasoning_Text ← Extract_Reasoning(Raw_Output) // "왜 이 페르소나인가?"

    RETURN Persona_Tag, Reasoning_Text
END ALGORITHM
```

### **Node 2. Candidate Retrieval (후보군 탐색)**

- **기능:** 페르소나에 매핑된 제품군으로 **Hard Filter**를 건 후, 고객 니즈 쿼리로 **Vector Search**를 수행한다.
- **Logic:**
    - `Filter:` `metadata.target_persona == classified_persona`
    - `Search:` 고객 니즈 임베딩 벡터와 제품 임베딩 간 코사인 유사도 계산
    - `Output:` Top-K Candidates List

```verilog
ALGORITHM Node2_Candidate_Retrieval
    INPUT: Persona_Tag, User_Profile, User_Logs
    OUTPUT: Candidate_List (Top-K)

    // 1. 임베딩 쿼리 생성
    User_Vector ← Embedding_Model(User_Logs.Keywords + User_Profile)
    
    // 2. 하드 필터링 & 벡터 검색 (Hard Filter)
    // 전체 DB가 아닌 해당 'Persona_Tag'를 가진 문서 집합 내에서만 검색
    Candidate_List ← VectorDB.SimilaritySearch(
        Query=User_Vector, 
        Top_K=5, 
        Filter={ "Target_Persona": Persona_Tag } // 메타데이터 필터 적용
    )
    
    RETURN Candidate_List
END ALGORITHM
```

### **Node 3.** Recommender (상품 선정 노드)

- **기능:** 단순 유사도만으로는 부족한 비즈니스 맥락을 반영하기 위해 가중치 점수 모델을 적용한다.
- **점수 function:** `Total = (유사도 × 0.5) + (행동점수 × 0.3) + (컨텍스트 × 0.2)`
    - 행동점수: 최근 본 상품, 체류 시간, 장바구니 데이터 반영
    - 컨텍스트: 날씨, 시즌, 기념일 적합성
- **Output:** 선정된 상품 ID(selectedProduct.productId) + 선정 이유(decisionReason)

```verilog
ALGORITHM Node3_Recommender
    INPUT: Candidate_List, User_Logs, Context
    OUTPUT: Best_Product_ID, Score_Breakdown

    Ranking_List ← []

    FOR EACH Product IN Candidate_List:
        // 1. 유사도 점수 (비중 50%)
        // 단순 임베딩/키워드 유사도
        Sim_Score ← Product.Similarity * 0.50
        
        // 2. 행동 점수 (비중 30%)
        // 구성: 최근 본 상품 + 체류 시간 + 장바구니 데이터
        Raw_Action_Score ← Calculate_Action(Product, User_Logs.Recent_View, User_Logs.Dwell_Time, User_Logs.Cart)
        Action_Score     ← Raw_Action_Score * 0.30
        
        // 3. 컨텍스트 점수 (비중 20%)
        // 구성: 날씨 + 시즌 + 기념일 적합성
        Raw_Ctx_Score ← Check_Context(Product, Context.Weather, Context.Season, Context.Anniversary)
        Ctx_Score     ← Raw_Ctx_Score * 0.20
        
        // 총점 계산
        Total_Score ← (Sim_Score + Action_Score + Ctx_Score)
        
        // 설명 가능한 추천 근거(Breakdown) 저장
        Score_Details ← {Sim: Sim_Score, Action: Action_Score, Ctx: Ctx_Score}
        Ranking_List.Append({Product, Total_Score, Score_Details})

    // 최종 제품 선정 (점수 내림차순 정렬 후 1위)
    Top_Selection ← Sort_Descending(Ranking_List).First()
    
    // 결과물: Best Product ID + Score Breakdown
    RETURN Top_Selection.Product.ID, Top_Selection.Score_Details
END ALGORITHM
```

### **Node 4. Prompt Builder (프롬프트 합성)**

- **기능:** 생성에 필요한 모든 입력을 조립한다.
- **Process:**
    1. **RAG:** 제품 ID로 성분, 효능, 리뷰 요약 데이터 조회
    2. **Brand DB:** 브랜드 ID로 톤앤매너 로드
    3. 글자 수 제한 등 제약사항 로드
    4. **Synthesis:** 시스템 프롬프트 구성

```verilog
ALGORITHM Node4_Prompt_Builder
    INPUT: Best_Product_ID, Recommended_Reason, Brand_ID
    OUTPUT: System_Prompt, Fact_Data

    // 1. RAG: 제품 상세 정보 조회
    // 제품 ID로 성분, 효능, 리뷰 요약 데이터를 구체적으로 조회함
    Fact_Data ← VectorDB.Search(
        Key=Best_Product_ID, 
        Target_Fields=["Ingredients", "Efficacy", "Review_Summary"]
    )
    
    // 2. Brand DB: 톤앤매너 및 제약사항 로드
    // 톤앤매너와 함께 '글자 수 제한(Constraints)'을 함께 로드
    Brand_Context ← SQL_Execute(
        "SELECT Tone_Manner, Char_Limit, Forbidden_Words FROM Brand_Table WHERE ID=?", 
        Brand_ID
    )
    
    // 3. Synthesis: 시스템 프롬프트 구성 (기능 반영)
    // 모든 입력(제품 정보, 브랜드 Rule/제약, 추천 사유)을 템플릿에 조립
    System_Prompt ← Template_Bind(
        Product_Info=Fact_Data,               // 성분, 효능, 리뷰
        Brand_Rule=Brand_Context,             // 톤앤매너 + 제약사항
        Reason=Recommended_Reason             // Node 3에서 넘어온 추천 근거
    )
    
    // 검증 단계(Node 6)에서 팩트 체크를 위해 Fact_Data도 함께 반환
    RETURN System_Prompt, Fact_Data
END ALGORITHM
```

### **Node 5.** CopyWriter (카피라이팅 노드)

- **기능:** 동일한 입력을 바탕으로 전략이 다른 3종의 메시지를 주기적으로 생성한다.
    1. **효능/기술 강조형:** 스펙과 Fact 중심
    2. **사용 상황 공감형:** 고객의 고민과 해결 상황 묘사
    3. **혜택/실용 정보형:** 가격, 증정, 사용 팁 중심

```verilog
ALGORITHM Node5_CopyWriter
    INPUT: System_Prompt
    OUTPUT: Draft_Messages[3]

    Draft_Messages ← []

    // 3가지 전략별 상세 지침 정의
    Strategy_Map ← [
        // 1. 효능/기술 강조형: 스펙과 Fact 중심
        {Type: "Tech_Spec",   Instruction: "Focus strictly on Product Specs and factual evidence."},
        
        // 2. 사용 상황 공감형: 고객의 고민과 해결 상황 묘사
        {Type: "Empathy",     Instruction: "Describe user pain points and how the product solves the situation."},
        
        // 3. 혜택/실용 정보형: 가격, 증정, 사용 팁 중심
        {Type: "Benefit_Tip", Instruction: "Highlight Price, Gifts, and practical Usage Tips."}
    ]

    // 3가지 전략 병렬 수행
    PARALLEL FOR Strategy IN Strategy_Map:
        
        // 시스템 프롬프트에 전략별 구체적 지침을 더해 생성
        Draft ← LLM_Generate(
            Input=System_Prompt, 
            Specific_Instruction=Strategy.Instruction
        )
        
        // 결과 저장 (유형과 내용을 함께 저장)
        Draft_Messages.Append({Type: Strategy.Type, Content: Draft})
        
    RETURN Draft_Messages
END ALGORITHM
```

### **Node 6. Compliance Guardian (**가드레일 검수 노드**)**

- **기능:** 생성된 메시지의 품질과 규칙 준수 여부를 검사하고, 실패 시 LLM이 구체적인 수정 지시를 생성하여 다시 생성 노드로 보낸다. (최대 5회 제한으로 무한 루프 방지)
- **검수 step:**
    1. **제약 조건 검수:** 글자 수(40/350자) 체크
    2. **톤앤매너 검수:** Gemini 3.0 Flash를 활용해 "톤앤매너 적합성" 판단
- **flow control:**
    - Pass → Node 7 (Final Output)
    - Fail → retry_count < 5 이면 Feedback 생성 후 Node 5로 복귀

```verilog
ALGORITHM Node6_Compliance_Guardian
    INPUT: Draft_Messages, Product_Fact_Data (from Node 4)
    OUTPUT: Validated_Messages

    Retry_Count ← 0
    
    WHILE Retry_Count < 5:
        Violations ← []

        // 1. Syntactic Check
        Violations.Add( Check_Length(Draft_Messages) )
        Violations.Add( Check_Forbidden_Words(Draft_Messages) )

        // 2. Semantic Check
        Violations.Add( Check_Hallucination(Draft_Messages, Source=Product_Fact_Data) )

        // 3. 통과 여부 결정
        IF Violations IS EMPTY:
            RETURN Draft_Messages
        
        // 4. 재수정 요청
        Feedback ← Generate_Feedback(Violations)
        Draft_Messages ← CopyWriter.Regenerate(Feedback)
        Retry_Count++

    RETURN Manual_Review_Request()
END ALGORITHM
```

### **Node 7. Final Suggestion Output (최종 출력 노드)**

- **기능:** 최종 결과~~를~~ 웹 UI로 전송하고 마케터의 사용 기록을 데이터화하여 시스템이 스스로 발전한다.
- **RLHF Data:** 마케터가 선택한 결과를 DB에 저장하여 선호도 데이터를 구축한다. 그리고 이를 추후 모델 튜닝 데이터로 활용한다.

```verilog
ALGORITHM Node7_Final_Suggestion_Output
    INPUT: Validated_Messages, Marketer_Selection(Event)
    OUTPUT: Display_UI, DB_Log

    // 1. 마케터에게 1~3종 제안 표시
    Display_UI(Validated_Messages)

    // 2. 마케터의 선택 대기 및 로깅
    Selected_Msg ← Wait_For_User_Action()
    
    // 3. 데이터 선순환을 위한 로그 적재
    Log_Data ← {
        "User_Context": User_Profile,
        "Predicted_Persona": Persona_Type,
        "Selected_Strategy": Selected_Msg.Strategy,
        "Result": "Success"
    }
    DB_Insert("Feedback_Logs", Log_Data)
    
    RETURN "Process Complete"
END ALGORITHM
```

<br>

# 4. 개발 단계별 주요 작업 내용

### [Phase 1] 데이터 수집 및 지식 베이스(KB) 구축

AI가 참조할 '제품'과 '브랜드' 데이터를 준비하는 단계이다.

1. **웹 크롤링**
    - `Selenium`과 `BeautifulSoup`을 활용하여 아모레몰(amoremall.com)의 제품 상세 페이지, 성분, 리뷰, 가격 정보를 수집한다.
2. **데이터 전처리 및 임베딩**
    - 수집된 텍스트를 의미 단위(Chunk)로 분할하고, OpenAI Embedding 모델 등을 사용하여 벡터화한 뒤 **Vector DB**에 적재한다.
    - 브랜드별 금기어, 권장 표현, 톤앤매너 가이드를 정리하여 **RDB** 또는 JSON 형태로 구축한다.

### [Phase 2] LangGraph 기반 에이전트 로직 구현

기획서 2-2의 5단계 흐름을 실제 코드로 구현하는 핵심 단계이다.

1. **Node(기능) 정의:** 각 기능을 수행할 함수를 LangGraph의 Node로 구현한다.
    - **Persona Classifier**: 고객 로그 입력 → LLM 분석 → 페르소나 도출
    - **Candidate Retrieval**: 페르소나 → Vector DB 검색 → Top-K 후보 추출
    - **Recommender**: Top-K 후보 + 고객 상황 → Chain-of-Thought 프롬프팅 → 최적 제품 1종 선정
    - **Prompt Builder + CopyWriter:** 제품 정보 + 브랜드 톤 → 메시지 3종 생성
    - **Compliance Guardian + Final Suggestion Output**: 생성된 메시지 검수 (글자 수, 톤 적합성 판단)
2. **Edge(흐름) 및 조건부 분기 설정**
    - Compliance Guardian 결과가 'Fail'일 경우 CopyWriter로 되돌아가는 조건 분기기를 설정한다. (재시도 횟수 `retry_count` 증가 로직 포함)
    - 검수 통과 시 종료 상태로 전이한다.

### [Phase 3] 웹 인터페이스 개발 및 통합

사용자(마케터)가 에이전트를 활용할 수 있는 UI를 구축한다.

1. **UI 구성**
    - 고객 리스트 선택 화면, 페르소나 분석 결과 표시 패널, 최종 메시지 카드 뷰 등을 구성한다.
2. **API 연동**
    - 웹 프론트엔드에서 생성 버튼 클릭 시 LangGraph의 `app.invoke(inputs)`를 호출하고 실시간 스트리밍 또는 최종 상태 반환값을 받아 화면에 렌더링한다.
3. **피드백 루프 구축**
    - 마케터가 최종 선택한 메시지 데이터를 별도 로그 파일이나 DB에 저장하는 기능을 구현하여 추후 프롬프트 개선에 활용한다.