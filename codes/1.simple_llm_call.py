from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# single one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatOpenAI()

parser = StrOutputParser()

# chain : prompt -> model -> parser
chain = prompt | model | parser


# run it
result = chain.invoke({'question':'What is the capital of Peru?'})
print(result)