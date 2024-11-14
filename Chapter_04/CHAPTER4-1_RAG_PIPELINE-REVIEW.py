
import os
os.environ['USER_AGENT'] = 'RAGUserAgent'
from langchain_community.document_loaders import WebBaseLoader
import bs4
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker


# OpenAI Setup
openai.api_key = os.getenv('OPENAI_API_KEY')




#### INDEXING ####
# Load Documents
loader = WebBaseLoader(
    web_paths=("https://kbourne.github.io/chapter1.html",), 
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split
text_splitter = SemanticChunker(OpenAIEmbeddings())
splits = text_splitter.split_documents(docs)


# Embed
vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()




#### RETRIEVAL and GENERATION ####


# Prompt - ignore LangSmith warning, you will not need langsmith for this coding exercise
prompt = hub.pull("jclemens24/rag-prompt")

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# Chain it all together with LangChain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Question - run the chain
resp = rag_chain.invoke("What are the advantages of using RAG?")
print("resp:", resp)



