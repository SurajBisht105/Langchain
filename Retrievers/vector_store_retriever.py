from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
# step1:    Your source documents
documents = [
    Document(page_content="Langchain helps developers build LLM applications easily"),                                                           
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectores."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# step2:    Initialize embedding model

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# step3:    Create Chroma vectore store in memory

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding = embedding_model,
    collection_name="my_collection"
)

retriever = vectorstore.as_retriever(kwargs={'k': 2})

query = "What is Chroma used for ?"

result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)