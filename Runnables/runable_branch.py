from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableBranch,RunnableSequence,RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following \n {text}",
    input_variables=['text']
)

report_gen_chain =RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>100, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))