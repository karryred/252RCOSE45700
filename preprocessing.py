import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- [3단계] 새로 추가된 라이브러리 ---
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# ------------------------------------

# ---- 0. 환경 변수 로드 ----
# .env 파일에서 API 키를 로드합니다.
load_dotenv() 

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("⚠️ OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
    print("   스크립트를 중지합니다. .env 파일을 확인해주세요.")
    exit() # 키가 없으면 3단계에서 무조건 실패하므로 중지
else:
    print("OPENAI_API_KEY 로드 완료.")

# ---- 1. 데이터 로드 (Load) ----
DATA_PATH = "./data_source"
print(f"\n--- 1단계 시작: '{DATA_PATH}' 폴더 로딩 ---")

pattern = os.path.join(DATA_PATH, "*.docx")
file_paths = glob.glob(pattern)

if not file_paths:
    print(f"⚠️ {DATA_PATH}에서 .docx 파일을 찾지 못했습니다. data 폴더 경로와 확장자를 확인하세요.")
    docs = []
else:
    docs = []
    print(f"총 {len(file_paths)}개의 .docx 파일을 찾았습니다.")
    for path in file_paths:
        try:
            print(f"- 로딩 중: {os.path.basename(path)}")
            loader = Docx2txtLoader(path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"  ❌ {path} 로딩 중 에러: {e}")

print(f"\n총 {len(docs)}개의 문서(Document)가 로드되었습니다.")

# ---- 2. 데이터 분할 (Split) ----
if docs: # 로드된 문서가 있을 때만 분할
    print("\n--- 2단계 시작: 문서 분할 ---")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)

    print(f"총 {len(chunks)}개의 텍스트 조각(Chunk)이 생성되었습니다.")

    if chunks:
        print("\n--- 첫 번째 조각 샘플 ---")
        print(chunks[0].page_content[:200])
        print(f"출처 (Metadata): {chunks[0].metadata}")
    else:
        print("⚠️ .docx 파일 내용은 있으나 텍스트 조각이 생성되지 않았습니다.")
else:
    print("⚠️ 로드된 문서가 없어 2단계(분할)를 건너뜁니다.")
    chunks = [] # 3단계에서 에러나지 않도록 빈 리스트로 초기화

# ---- 3. 임베딩 및 벡터 스토어 저장 (Embed & Store) ----
if chunks:
    print("\n--- 3단계 시작: 임베딩 및 벡터 스토어 생성 ---")

    # 3-1. 임베딩 모델 초기화 (OpenAI API 키 사용)
    # 한국어에 강점이 있는 text-embedding-3-small 모델 사용
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("OpenAI 임베딩 모델 ('text-embedding-3-small')을 초기화했습니다.")

    # 3-2. FAISS 벡터 스토어 생성
    # chunks(문서 조각)와 임베딩 모델을 사용해 벡터 DB 생성
    print("텍스트 조각들을 FAISS 벡터 스토어로 변환 중... (OpenAI API 호출 중...)")
    try:
        vectorstore = FAISS.from_documents(
            documents=chunks, 
            embedding=embeddings
        )
        print("FAISS 벡터 스토어 생성 완료!")

        # 3-3. 벡터 스토어 로컬에 저장
        # 나중에 재사용할 수 있도록 파일로 저장
        VECTORSTORE_PATH = "faiss_index"
        vectorstore.save_local(VECTORSTORE_PATH)
        print(f"벡터 스토어를 로컬 폴더 '{VECTORSTORE_PATH}'에 성공적으로 저장했습니다.")

    except Exception as e:
        print(f"\n[!!! 오류] 3단계 임베딩 또는 벡터 스토어 생성/저장 중 오류 발생: {e}")
        print("  -> OPENAI_API_KEY가 올바른지, 잔액이 충분한지 확인하세요.")
        print("  -> 'langchain-openai', 'faiss-cpu' 라이브러리가 설치되었는지 확인하세요.")

else:
    print("\n--- 3단계 건너뜀: 처리할 텍스트 조각(chunks)이 없습니다. ---")

print("\n--- 모든 전처리 단계(1-3) 완료 ---")