import os
import gradio as gr
from dotenv import load_dotenv

# --- RAG 핵심 라이브러리 임포트 ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. RAG 리소스 초기화 ---

print("RAG 챗봇 리소스 초기화 중...")

# 1-1. 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")

# 1-2. 모델 초기화 (LLM, Embeddings)
# (Gradio는 앱 실행 내내 살아있으므로 전역 변수로 로드)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 1-3. FAISS 벡터 스토어 로드
VECTORSTORE_PATH = "faiss_index"
if not os.path.exists(VECTORSTORE_PATH):
    raise FileNotFoundError(f"'{VECTORSTORE_PATH}' 폴더를 찾을 수 없습니다.")

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH, 
    embeddings,
    allow_dangerous_deserialization=True
)

# 1-4. 리트리버 생성
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

print("LLM, Retriever(FAISS) 로드 완료.")

# --- 2. (신규) RAG 모드별 체인 생성 함수 ---

# 검색된 문서를 프롬프트용 문자열로 포맷팅
def format_docs_with_sources(docs):
    formatted_list = []
    for doc in docs:
        source_filename = os.path.basename(doc.metadata.get('source', '알 수 없음'))
        formatted_entry = (
            f"Source: {source_filename}\n"
            f"Content: {doc.page_content}"
        )
        formatted_list.append(formatted_entry)
    return "\n\n---\n\n".join(formatted_list)

# 2-1. [RAG On] RAG + 대화 기록 체인 생성 (프롬프트 수정됨)
def create_chain_with_kb(retriever, llm):
    """
    RAG(검색)와 대화 기록을 모두 사용하는 체인을 생성합니다.
    [수정됨] RAG 문맥이 질문과 관련 없으면, 일반 지식으로 답변하도록 프롬프트를 수정했습니다.
    """
    
    # [수정] RAG용 프롬프트: AI가 상황에 맞게 RAG/일반 답변을 선택하도록 지시
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 친절한 AI 어시스턴트입니다.
아래 [문맥]은 사용자의 과제물(문서)에서 검색된 내용입니다.

1.  먼저 [질문]이 [문맥]의 내용과 관련이 있는지 확인하세요.
2.  **만약 [문맥]과 관련이 있다면:** [문맥]을 바탕으로 [질문]에 답변해 주세요. 그리고 답변 마지막에 (출처: [Source 파일명])을 꼭 명시하세요.
3.  **만약 [문맥]과 관련이 없거나** [질문]이 일반적인 대화(인사, 농담, 날씨 등)라면: [문맥]을 무시하고, 당신의 일반 지식을 사용해 답변하세요. 이때는 출처를 명시하지 마세요.

---
[문맥]
{context}
"""),
        MessagesPlaceholder(variable_name="chat_history"), # 대화 기록
        ("human", "{input}"), # 현재 질문
    ])
    
    # 문맥(Context) 검색 체인 (기존과 동일)
    context_chain = (
        (lambda x: x["input"]) 
        | retriever 
        | format_docs_with_sources
    )
    
    # RAG 체인 구성 (기존과 동일)
    rag_chain = (
        {
            "context": context_chain,
            "chat_history": lambda x: x["chat_history"],
            "input": lambda x: x["input"]
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# 2-2. [RAG Off] 일반 대화 + 대화 기록 체인 생성
def create_chain_without_kb(llm):
    """RAG(검색) 없이, 대화 기록만 사용하는 일반 채팅 체인을 생성합니다."""
    
    # 일반 대화용 프롬프트
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 AI 어시스턴트입니다."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # 일반 채팅 체인
    chat_chain = (
        chat_prompt
        | llm
        | StrOutputParser()
    )
    return chat_chain

# --- 3. (변경) Gradio 채팅 응답 함수 ---
# (Gradio는 리소스를 전역(Global)에 두고, 함수는 호출될 때마다 실행)

# 체인을 미리 생성 (RAG On/Off 모두)
rag_chain_with_kb = create_chain_with_kb(retriever, llm)
rag_chain_without_kb = create_chain_without_kb(llm)

def chat_response_generator(message, history, use_kb):
    """
    Gradio ChatInterface가 호출할 메인 함수.
    - message: 사용자의 현재 입력 (문자열)
    - history: 이전 대화 기록 (Gradio 형식: [[user, ai], [user, ai], ...])
    - use_kb: 'additional_inputs'로 추가한 체크박스의 값 (True/False)
    """
    
    # 1. (신규) Gradio의 'history'를 LangChain의 'Message' 형식으로 변환
    langchain_history = []
    for user_msg, ai_msg in history:
        langchain_history.append(HumanMessage(content=user_msg))
        langchain_history.append(AIMessage(content=ai_msg))

    # 2. (신규) RAG 모드(use_kb)에 따라 사용할 체인 선택
    if use_kb:
        chain = rag_chain_with_kb
    else:
        chain = rag_chain_without_kb

    # 3. (신규) 체인에 입력할 딕셔너리 생성
    input_dict = {
        "input": message,
        "chat_history": langchain_history
    }
    
    # 4. (기존 로직) 체인 스트리밍 실행
    response_stream = chain.stream(input_dict)
    
    partial_response = ""
    for chunk in response_stream:
        partial_response += chunk
        yield partial_response # 스트리밍으로 응답 반환

# --- 4. (변경) Gradio 인터페이스 실행 ---

# gr.ChatInterface: 채팅 UI를 즉시 생성
# [핵심] 'additional_inputs'에 체크박스를 추가
gr.ChatInterface(
    fn=chat_response_generator, # 응답을 생성할 함수
    
    # [신규] RAG 모드 On/Off 체크박스 추가
    additional_inputs=[
        gr.Checkbox(
            label="과제물(RAG) 검색 사용", 
            value=True, # 기본값 True
            info="체크 시: 과제물 내용을 검색하여 답변합니다.\n체크 해제 시: 일반 AI 챗봇처럼 대화합니다."
        )
    ],
    
    title="🤖 나의 과제물 RAG 챗봇 (V2)",
    description="LangChain과 Gradio로 구축한 RAG 챗봇 (RAG On/Off, 대화기록 기능 탑재)",
    
    # # [수정] "리스트의 리스트" 형식으로 변경
    # # 각 예시가 [질문(message), RAG체크박스(use_kb)]의 짝을 이룸
    # examples=[
    #     ["[여기에 예시 질문 1 입력]", True], # 예시1 (RAG 켠 상태)
    #     ["[여기에 예시 질문 2 입력]", False]  # 예시2 (RAG 끈 상태)
    # ],
    
    # cache_examples=False 
).launch(share=True)