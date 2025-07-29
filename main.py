from app.retriever import create_retriever
from app.rag_chain import create_rag_chain

def main():
    """
    Creates the full RAG pipeline, generates an answer, and prints token usage.
    """
    retriever, client = None, None
    try:
        retriever, client = create_retriever()
        rag_chain = create_rag_chain(retriever)

        query = "Who is chad walding?"

        print(f"--- Querying the RAG chain for: '{query}' ---")

        # The chain now returns an AIMessage object, not a string
        response = rag_chain.invoke(query)

        # Extract the answer from the 'content' attribute
        answer = response.content
        
        # Extract token usage from the 'response_metadata'
        token_usage = response.response_metadata.get('token_usage', {})
        prompt_tokens = token_usage.get('prompt_tokens', 0)
        completion_tokens = token_usage.get('completion_tokens', 0)
        total_tokens = token_usage.get('total_tokens', 0)

        print("\n--- Generated Answer ---")
        print(answer)
        
        print("\n--- Token Usage ---")
        print(f"Prompt Tokens: {prompt_tokens}")
        print(f"Completion Tokens: {completion_tokens}")
        print(f"Total Tokens: {total_tokens}")

    finally:
        if client and client.is_connected():
            client.close()
            print("\n--- Weaviate connection closed ---")


if __name__ == "__main__":
    main()