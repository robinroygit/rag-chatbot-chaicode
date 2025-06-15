# requirements.txt
# langchain, openai, pinecone-client, python-dotenv

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI

# Load environment variables
load_dotenv()

# Step 1: Load ChaiCode Docs
print("Loading documentation...")
loader = WebBaseLoader(
     urls = [
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
    # header_template={"User-Agent": os.getenv("USER_AGENT")}
)
docs = loader.load()


# Step 2: Chunking
print("Chunking documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(docs)


# Step 3: Embeddings
print("Creating embeddings...")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Step 4: Store in Pinecone
print("Storing vectors in Pinecone...")
import pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

vector_store = PineconeVectorStore.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    index_name="chai-docs-index"
)

print("Indexing complete. Ready to chat!")

# # Step 5: Chat loop
# client = OpenAI()

# while True:
#     query = input("\nAsk ChaiCode Docs > ")
#     if query.lower() in ["exit", "quit"]:
#         break

#     search_results = vector_store.similarity_search(query, k=4)
#     context = "\n\n".join([
#         f"Page: {doc.page_content}\nURL: {doc.metadata.get('source', 'N/A')}"
#         for doc in search_results
#     ])

#     SYSTEM_PROMPT = f"""
#     You are a helpful assistant for ChaiCode documentation.
#     You must only answer based on the provided context.
#     Always include the reference URL so the user can read more.

#     Context:
#     {context}
#     """

#     chat_completion = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": query}
#         ]
#     )

#     print("\n🤖", chat_completion.choices[0].message.content)
