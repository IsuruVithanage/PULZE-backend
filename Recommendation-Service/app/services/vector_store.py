# app/services/vector_store.py
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

pc = Pinecone(api_key=settings.PINECONE_API_KEY)

# Initialize the embedding model once
embedding_model = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'} # Use CPU for embedding
)

def get_or_create_vector_store() -> PineconeVectorStore:
    """
    Initializes and returns a PineconeVectorStore instance.
    Creates the index if it doesn't exist.
    """
    # Check if the index already exists
    if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {settings.PINECONE_INDEX_NAME}")
        # Create a new index
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.EMBEDDING_DIMENSION,
            metric="cosine",  # Common choice for semantic similarity
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1' # Choose a region
            )
        )
    else:
        print(f"Connecting to existing Pinecone index: {settings.PINECONE_INDEX_NAME}")

    # Initialize the PineconeVectorStore
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embedding_model
    )
    return vectorstore