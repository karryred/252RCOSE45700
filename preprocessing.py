import os
import glob
import bs4 # HTML 태그 정제용 (BeautifulSoup)
from dotenv import load_dotenv

# --- 문서 로더 임포트 ---
from langchain_community.document_loaders import Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---- 0. 환경 변수 로드 ----
load_dotenv() 

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("⚠️ OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
    exit()
else:
    print("✅ OPENAI_API_KEY 로드 완료.")

# ---- 1. 데이터 로드 (Load) ----
print("\n--- 1단계 시작: 데이터 소스 로딩 ---")

# [1-1] 로컬 Docx 파일 로드
DATA_PATH = "./data_source"
pattern = os.path.join(DATA_PATH, "*.docx")
file_paths = glob.glob(pattern)

file_docs = []

if file_paths:
    print(f"📂 로컬: '{DATA_PATH}'에서 {len(file_paths)}개의 .docx 파일 발견")
    for path in file_paths:
        try:
            loader = Docx2txtLoader(path)
            file_docs.extend(loader.load())
            print(f"   - [파일] 로드 성공: {os.path.basename(path)}")
        except Exception as e:
            print(f"   - [파일] ❌ 에러: {os.path.basename(path)} ({e})")
else:
    print(f"📂 로컬: '{DATA_PATH}'에 .docx 파일이 없습니다. (건너뜀)")


# [1-2] 웹사이트 URL 로드 (수정됨)
WEB_URL = "https://ko.wikipedia.org/wiki/%EB%8D%B0%EC%9D%B4%ED%81%AC%EC%8A%A4%ED%8A%B8%EB%9D%BC_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98"

web_docs = []
try:
    print(f"🌐 웹: URL 로딩 시작")
    loader = WebBaseLoader(
        web_paths=(WEB_URL,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "div", attrs={"id": ["mw-content-text", "bodyContent"]}
            )
        ),
    )
    web_docs = loader.load()

    # =========== [여기만 추가하시면 됩니다!] ===========
    if web_docs:
        for doc in web_docs:
            # 1. 깨진 URL(%EC...)을 가져와서
            original_source = doc.metadata.get('source', '')
            # 2. 한글로 복구(Decode)하고
            decoded_source = unquote(original_source)
            # 3. 다시 덮어씌웁니다.
            doc.metadata['source'] = decoded_source
            
        print(f"   - [변환 완료] 출처가 한글로 변경됨: {web_docs[0].metadata['source']}")
    # =================================================
        
except Exception as e:
    print(f"   - [웹] ❌ 에러 발생: {e}")


# [1-3] 데이터 합치기
all_docs = file_docs + web_docs
print(f"\n📊 총 {len(all_docs)}개의 문서(Document)가 준비되었습니다.")


# ---- 2. 데이터 분할 (Split) ----
if all_docs:
    print("\n--- 2단계 시작: 문서 분할 ---")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_documents(all_docs)

    print(f"🧩 총 {len(chunks)}개의 텍스트 조각(Chunk) 생성 완료")

    if chunks:
        print(f"   (샘플) 첫 번째 조각 출처: {chunks[0].metadata.get('source')}")
        # 샘플 내용을 조금 출력해서 잘 가져왔는지 확인
        print(f"   (샘플 내용) {chunks[0].page_content[:100]}...")
else:
    print("⚠️ 로드된 데이터가 전혀 없어 종료합니다.")
    exit()

# ---- 3. 임베딩 및 벡터 스토어 저장 (Embed & Store) ----
if chunks:
    print("\n--- 3단계 시작: 임베딩 및 벡터 스토어 생성 ---")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("🤖 OpenAI 임베딩 모델 초기화 완료")

    print("📦 FAISS 벡터 스토어 생성 중... (API 호출)")
    try:
        vectorstore = FAISS.from_documents(
            documents=chunks, 
            embedding=embeddings
        )
        
        VECTORSTORE_PATH = "faiss_index"
        vectorstore.save_local(VECTORSTORE_PATH)
        print(f"✅ 저장 완료! 벡터 스토어가 '{VECTORSTORE_PATH}' 폴더에 저장되었습니다.")

    except Exception as e:
        print(f"\n❌ [오류] 임베딩/저장 실패: {e}")
else:
    print("❌ 처리할 텍스트 조각이 없습니다.")