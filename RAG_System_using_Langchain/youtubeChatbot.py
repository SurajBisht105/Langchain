# ! pip install -q youtube-transcript-api langchain-community langchain-openai faiss-cpu tiktoken python-dotenv


from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()



video_id = "Gfr50f6ZBvo"   #   only the ID, not full URL
try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])

    transcript = " ".join(chunk.text for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")



splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
# print(len(chunks))



embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.from_documents(chunks, embeddings)
 


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
retriever.invoke("What is deepmind")



llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
""",
input_variables= ['context', 'question']
)



question = "is this topic of aliens discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)


context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text ,"question":question})



answer= llm.invoke(final_prompt)
print(answer)



def format_docs(retriever_docs):
    context_text = "\n\n".join(doc.page_content for doc in retriever_docs)  
    return context_text

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough() 
})

parallel_chain.invoke("who is denis")

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

main_chain.invoke("can you summarize the video")