import os
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Load environment variables from .env
load_dotenv()

mavericks_profiles_file_path = os.path.join(os.path.pardir, "assets", "profiles.json")
chroma_db_directory = os.path.join(os.path.pardir, "databases", "chroma_db")

embeddings = OpenAIEmbeddings(
    check_embedding_ctx_length=False,
    model="text-embedding-nomic-embed-text-v1.5",
    base_url="http://127.0.0.1:1234/v1"
)  

model = ChatOpenAI(
    model="llama-3.2-1b-instruct", 
    base_url="http://127.0.0.1:1234/v1"
)

def load_mavericks_profiles(path: str) -> dict:
    """
    Load the mavericks profiles from the given file path and return json.
    """
    profiles = []
    with open(path, 'r', encoding='utf-8') as file:
        profiles = json.load(file)
    return profiles

def mavericks_profiles_to_documents(profiles: list) -> list[Document]:
    """
    Convert the mavericks profiles to a list of text documents.
    """
    documents: list[Document] = []
    for profile in profiles:
        page_content = (
            f"Name = {profile['name']}\n" +
            f"Email = {profile['email']}\n" +
            f"Region = {profile['region']}\n" +
            f"Designation = {profile['designation']}\n" +
            f"Category Expertise = {profile['categoryExpertise'] if 'categoryExpertise' in profile else ''}\n" +
            f"Engagement Status = {profile['engagementStatus'] if 'engagementStatus' in profile else ''}\n" +
            f"Industry = {profile['industry'] if 'industry' in profile else ''}\n" +
            f"Summary = {profile['summary'] if 'summary' in profile else ''}\n" +
            f"Skills = {', '.join(profile['skills'] if 'skills' in profile else [])}\n"
        )
        document = Document(page_content=page_content)
        document.metadata = { "email": profile['email'] }
        documents.append(document)
    return documents

if not os.path.exists(chroma_db_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Load the mavericks profiles
    mavericks_profiles = load_mavericks_profiles(mavericks_profiles_file_path)

    # Convert the mavericks profiles to documents 
    documents = mavericks_profiles_to_documents(mavericks_profiles)

    # Display information about the split documents
    print("\n--- Document Profiles Information ---")
    print(f"Number of document profiles: {len(documents)}")
    print(f"Sample document profile:\n{documents[0].page_content}\n")

    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        documents, embeddings, persist_directory=chroma_db_directory)
    print("\n--- Finished creating vector store ---")  

else:
    # Load the existing vector store with the embedding function
    db = Chroma(persist_directory=chroma_db_directory,
            embedding_function=embeddings)


query = "What is the email of Kathryn?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "score_threshold": 0.5},
)

relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata}\n")