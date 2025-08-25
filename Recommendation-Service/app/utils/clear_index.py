# clear_index.py
import os
from pinecone import Pinecone
from dotenv import load_dotenv
import time

def clear_pinecone_index():
    """
    Connects to Pinecone and deletes all vectors in the specified index.
    """
    # Load environment variables from your .env file
    load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not pinecone_api_key or not index_name:
        print("Error: PINECONE_API_KEY or PINECONE_INDEX_NAME not found in .env file.")
        return

    print(f"Connecting to Pinecone to clear index '{index_name}'...")

    # WARNING: This action is irreversible.
    confirmation = input(f"Are you sure you want to delete ALL vectors in the index '{index_name}'? (yes/no): ")

    if confirmation.lower() != 'yes':
        print("Operation cancelled.")
        return

    try:
        pc = Pinecone(api_key=pinecone_api_key)

        if index_name not in pc.list_indexes().names():
            print(f"Index '{index_name}' does not exist. Nothing to clear.")
            return

        index = pc.Index(index_name)

        print("Deleting all vectors from the index...")
        index.delete(delete_all=True)

        # Deletion can take a moment. We wait until the index is empty.
        while True:
            stats = index.describe_index_stats()
            if stats.total_vector_count == 0:
                break
            print("Waiting for deletion to complete...")
            time.sleep(5)

        print("Successfully deleted all vectors.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    clear_pinecone_index()