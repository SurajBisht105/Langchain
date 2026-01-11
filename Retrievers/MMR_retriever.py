# MMR retriever (Maximal Marginal Relevance)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from dotenv import load_dotenv

load_dotenv()

docs= [
    Document(page_content="LangChain makes it easy to with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# step2:    Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# step3:    Create Chroma vectore store in memory
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model,
)

# Enable MMR in the retriever

retriever = vectorstore.as_retriever(
    search_types='mmr',                                    #  <--- This enables MMR
    search_kwargs={"k": 3, "lambda_mult": 0.5}    #  k = top results, lambda_mult = relevance-diversity balance
)

query = "What is langchain ?"

result = retriever.invoke(query)


for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)