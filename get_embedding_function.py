from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# load_dotenv()
# api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-multilingual-gemma2")
    embeddings = HuggingFaceEmbeddings()
    return embeddings
