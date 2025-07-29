import weaviate
from config import settings

def clear_weaviate_index():
    """
    Connects to Weaviate and deletes all data from the specified index.
    """
    try:
        # Connect to your local Weaviate instance
        client = weaviate.connect_to_local(
            host=settings.WEAVIATE_HOST,
            port=settings.WEAVIATE_PORT
        )

        index_name = settings.INDEX_NAME
        
        # Check if the collection exists before trying to delete
        if client.collections.exists(index_name):
            print(f"Index '{index_name}' exists. Deleting all objects...")
            # Delete the entire collection/class
            client.collections.delete(index_name)
            print(f"Successfully deleted index '{index_name}'.")
        else:
            print(f"Index '{index_name}' does not exist. Nothing to delete.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'client' in locals() and client.is_connected():
            client.close()

if __name__ == '__main__':
    # This will clear your Weaviate index before running the main script
    clear_weaviate_index()