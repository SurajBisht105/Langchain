from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import  load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Generate the tweet about {topic}",
    input_variables="topic"
)

prompt2=PromptTemplate(
    template="Generate the linkedin post about {topic}",
    input_variables="topic"
)

parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "post": RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print(result)