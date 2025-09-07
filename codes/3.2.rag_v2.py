# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv


# *********************************** We can trace anything using @traceable of Langsmith as different experiments***********************
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- LangSmith env (make sure these are set) ---
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...
# LANGCHAIN_PROJECT=pdf_rag_demo


from langsmith import traceable # <-- key import

os.environ['LANGCHAIN_PROJECT'] = 'RAG chatbot hi'

load_dotenv()

 
PDF_PATH = 'health_policy_doc.pdf'# <-- change to your PDF filename


# ---------- traced setup steps ----------

# 1) Load PDF
@traceable(name="load_pdf")
def load_pdf(path : str):
    loader = PyPDFLoader(PDF_PATH)
    return loader.load() # list[Document]

# 2) Chunk
@traceable(name= "split_documents")
def split_documents(docs, chunk_size=500, chunk_overlap = 50 ):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    return splitter.split_documents(docs)


# 3) Embed + Index
@traceable(name = "build_vectorstore")
def build_vectorstore(splits):
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    # FAISS.from_documents internally calls the embedding model:
    vs = FAISS.from_documents(splits, emb)
    ## retriever = vs.as_retriever(search_type = 'similarity', search_kwargs={'k':3})
    return vs

# You can also trace a “setup” umbrella span if you want:
@traceable(name = "setup_pipeline")
def setup_pipeline(pdf_path : str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs

# ---------- pipeline ----------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4) Prompts

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")]
                                          )

# merging the chunk of text extracted by retriever
def format_docs(docs): 
    return "\n".join(d.page_content for d in docs)

# Build the index under traced setup
vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 5) chain
parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question" : RunnablePassthrough()
})


chain = parallel | prompt | llm | StrOutputParser()

# ---------- run a query (also traced) ----------


# 6) Ask the questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
# What are the coverage limits and conditions for maternity benefits, including normal delivery and C-section?
q = input("\nQ: ")


# Give the visible run name + tags/metadata so it’s easy to find:
config = {
    "run_name" : "pdf_rag_query"
}

ans = chain.invoke(q, config=config)

print("\nA : ", ans)

