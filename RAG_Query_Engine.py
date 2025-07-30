from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("docs/", glob="**/*.txt")
docs = loader.load()

# OpenAI의 텍스트 임베딩 모델 사용
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectordb = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")

# SKT A.X 4.1 모델 호출
llm = OpenAI(model_name="ax-4.1", temperature=0)

# 벡터 스토어에서 상위 k개 문서 검색
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# RAG용 QA 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",        # "map_reduce", "refine" 등 상황에 맞게 선택
    retriever=retriever,
    return_source_documents=True
)

# 실제 질의
query = "슬로우쿼리에 대해 알려줘"
result = qa_chain({"query": query})

print("답변:", result["result"])
print("참조 문서:", [doc.metadata["source"] for doc in result["source_documents"]])
