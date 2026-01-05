from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


prompt=PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=['question','text']
)

parser = StrOutputParser()  

url = "https://www.amazon.in/OnePlus-Charcoal-Snapdragon%C2%AE-Personalised-Game-Changing/dp/B0FZSWZZW2/ref=sr_1_1_sspa?crid=SN43NL26M2RM&dib=eyJ2IjoiMSJ9.7QrksMjjZf_nzv7o4smFA6_ZhKAmq8G8RohGECVVgNckxLMOIPja295fe5qkTz-kCcZkfQslRV3yCn6IC-JedP6tVcQ0MQGl9io7qg39NZa_jbuqy666Nb8PZBadjZiZowNG12Ivv_WY7L0FkPSyRRPiGejWuGsse5ehA86nIph4YaPIjvUgWs2-StNpqpZPWqunqEF3ScvErWZS7TA2dRpq1ioMkDqm2URxH-OlXGw.xndDMbx798Ka238qNwjfV3Km_lzqaPLKNMdIBbpA1U4&dib_tag=se&keywords=mobile&qid=1767540679&sprefix=mobile%2Caps%2C455&sr=8-1-spons&aref=0wL81b5McE&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1"

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question':'what is the RAM of this model','text':docs[0].page_content}))