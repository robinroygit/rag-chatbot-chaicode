# pip install streamlit langchain langchain-openai beautifulsoup4 python-dotenv pinecone-client

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os

# Load environment variables
load_dotenv()

# Pinecone config
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "website-chat-index"
namespace = "chaicode-docs"

all_urls = [
        "https://docs.chaicode.com/youtube/getting-started/",
        "https://docs.chaicode.com/youtube/chai-aur-html/welcome/",
        "https://docs.chaicode.com/youtube/chai-aur-html/introduction/",
        "https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/",
        "https://docs.chaicode.com/youtube/chai-aur-html/html-tags/",
        "https://docs.chaicode.com/youtube/chai-aur-git/welcome/",
        "https://docs.chaicode.com/youtube/chai-aur-git/introduction/",
        "https://docs.chaicode.com/youtube/chai-aur-git/terminology/",
        "https://docs.chaicode.com/youtube/chai-aur-git/behind-the-scenes/",
        "https://docs.chaicode.com/youtube/chai-aur-git/branches/",
        "https://docs.chaicode.com/youtube/chai-aur-git/diff-stash-tags/",
        "https://docs.chaicode.com/youtube/chai-aur-git/managing-history/",
        "https://docs.chaicode.com/youtube/chai-aur-git/github/",
        "https://docs.chaicode.com/youtube/chai-aur-c/welcome/",
        "https://docs.chaicode.com/youtube/chai-aur-c/introduction/",
        "https://docs.chaicode.com/youtube/chai-aur-c/hello-world/",
        "https://docs.chaicode.com/youtube/chai-aur-c/variables-and-constants/",
        "https://docs.chaicode.com/youtube/chai-aur-c/data-types/",
        "https://docs.chaicode.com/youtube/chai-aur-c/operators/",
        "https://docs.chaicode.com/youtube/chai-aur-c/control-flow/",
        "https://docs.chaicode.com/youtube/chai-aur-c/loops/",
        "https://docs.chaicode.com/youtube/chai-aur-c/functions/",
        "https://docs.chaicode.com/youtube/chai-aur-django/welcome/",
        "https://docs.chaicode.com/youtube/chai-aur-django/getting-started/",
        "https://docs.chaicode.com/youtube/chai-aur-django/jinja-templates/",
        "https://docs.chaicode.com/youtube/chai-aur-django/tailwind/",
        "https://docs.chaicode.com/youtube/chai-aur-django/models/",
        "https://docs.chaicode.com/youtube/chai-aur-django/relationships-and-forms/",
        "https://docs.chaicode.com/youtube/chai-aur-sql/welcome/",
        "https://docs.chaicode.com/youtube/chai-aur-sql/introduction/",
        "https://docs.chaicode.com/youtube/chai-aur-sql/postgres/",
        "https://docs.chaicode.com/youtube/chai-aur-sql/normalization/",
        "https://docs.chaicode.com/youtube/chai-aur-sql/database-design-exercise/",
        "https://docs.chaicode.com/youtube/chai-aur-sql/joins-and-keys/",
        "https://docs.chaicode.com/youtube/chai-aur-sql/joins-exercise/",
        "https://docs.chaicode.com/youtube/chai-aur-devops/welcome/",
        "https://docs.chaicode.com/youtube/chai-aur-devops/setup-vpc/",
        "https://docs.chaicode.com/youtube/chai-aur-devops/setup-nginx/",
        "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-rate-limiting/",
        "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-ssl-setup/",
        "https://docs.chaicode.com/youtube/chai-aur-devops/node-nginx-vps/",
        "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-docker/",
        "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-vps/",
        "https://docs.chaicode.com/youtube/chai-aur-devops/node-logger/",
        ]

# System prompt
SYSTEM_PROMPT = f"""
You are a helpful AI assistant who answers user queries based on the available context
retrieved from website data, including page content and URLs.
You must only use the provided context and guide the user to the correct url for more details.
URL:
{all_urls}
"""

def get_vectorstore_from_url():
    urls = [
        "https://docs.chaicode.com/youtube/getting-started/",
        "https://docs.chaicode.com/youtube/chai-aur-html/welcome/",
        "https://docs.chaicode.com/youtube/chai-aur-html/introduction/",
        "https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/",
        "https://docs.chaicode.com/youtube/chai-aur-html/html-tags/",
        "https://docs.chaicode.com/youtube/chai-aur-git/welcome/",
       
    ]

    index = pc.Index(index_name)
    stats = index.describe_index_stats()

    if namespace not in stats.get("namespaces", {}):
        st.info("Uploading documents to Pinecone. This may take a moment...")
        loader = WebBaseLoader(urls)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()

        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name,
            namespace=namespace
        )
        st.success("Documents uploaded to Pinecone.")

    return PineconeVectorStore(
        index_name=index_name,
        embedding=OpenAIEmbeddings(),
        namespace=namespace
    )

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"{SYSTEM_PROMPT}\n\nContext:\n{{context}}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    return create_retrieval_chain(retriever_chain, create_stuff_documents_chain(llm, prompt))

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    rag_chain = get_conversational_rag_chain(retriever_chain)

    response = rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']


# Streamlit UI
st.set_page_config(page_title="Chat with ChaiCode Docs", page_icon="🤖")
st.title("Chat with ChaiCode Docs")

# Initialize vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_url()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am your AI assistant. How can I help you?")
    ]

user_query = st.chat_input("Ask something about ChaiCode Docs...")
if user_query:
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# Display conversation
for msg in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(msg, AIMessage) else "Human"):
        st.write(msg.content)
