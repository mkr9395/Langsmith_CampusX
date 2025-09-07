from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# adding it to a new project
os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'


load_dotenv()

prompt1 = PromptTemplate(
    template = 'generate a detailed report on {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a 5 pointer summary from the following text \n {text}.',
    input_variables = ['text']
)

model1 = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0.7)
model2 = ChatOpenAI(model = 'gpt-4o', temperature = 0.5)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

# add tags,metadate that will be logged
config = {
    'run_name' : 'sequential chain',
    'tags' : ['llm app', 'report generation', 'summarization'], 
    'metadata' : {'model1':'gpt-4o-mini', 'model1_temp':0.7,'parser':'StrOuputParser','model2':'gpt-4o', 'model1_temp':0.5}
}

result = chain.invoke({'topic':'How to learn AI for job security'}, config = config)

print(result)