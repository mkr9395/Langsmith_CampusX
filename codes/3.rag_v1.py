# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGCHAIN_PROJECT'] = 'RAG chatbot hi'

load_dotenv()

 
PDF_PATH = 'health_policy_doc.pdf'# <-- change to your PDF filename

# 1) Load PDF
loader=PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 50)
splits = splitter.split_documents(docs)

# 3) Embed + Index
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type = 'similarity', search_kwargs={'k':3})

# 4) Prompts
prompt = ChatPromptTemplate.from_messages([
    ('system', "Answer ONLY from the provided context. If not found, say you don't know."),
    ('human', "Question: {question}\n\nContext:\n{context}")]
                                           )

# 5) chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# merging the chunk of text extracted by retriever
def format_docs(docs): 
    return "\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question" : RunnablePassthrough()
})


chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask the questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
# What are the coverage limits and conditions for maternity benefits, including normal delivery and C-section?
q = input("\nQ: ")
ans = chain.invoke(q.strip())

print("\nA : ", ans)

