from typing import List
from langchain_core.documents import Document
# from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState

from app.retriever import create_retriever
from config import settings

# --- 1. Update the state to include token counts ---
class GraphState(MessagesState):
    documents: List[Document]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

# --- 2. The retrieve_documents node remains the same ---

def retrieve_documents(state: GraphState) -> dict:
    """
    Node to retrieve documents. The input is the full message history.
    We'll use the last human message as the query.
    """
    print("---NODE: RETRIEVE DOCUMENTS---")
    last_message = state['messages'][-1]
    question = last_message.content
    
    retriever, client = create_retriever()
    documents = retriever.invoke(question)
    client.close()
    
    return {"documents": documents}


def generate_answer(state: GraphState) -> dict:
    """
    Node to generate an answer and extract token usage.
    """
    print("---NODE: GENERATE ANSWER---")
    question = state['messages'][-1].content
    documents = state["documents"]

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context. 
        Also consider the chat history to understand the question properly.
        
        Chat History:
        {chat_history}

        Context:
        {context}

        Question: {question}"""
    )

    llm = ChatOpenAI(
        model_name=settings.OPENAI_MODEL_NAME,
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=0.7
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        prompt
        | llm
    )
    
    generation = rag_chain.invoke({
        "question": question, 
        "context": documents,
        "chat_history": state['messages']
    })
    
    # --- 3. Extract token usage and return it with the message ---
    token_usage = generation.response_metadata.get('token_usage', {})
    
    return {
        "messages": [generation],
        "prompt_tokens": token_usage.get('prompt_tokens', 0),
        "completion_tokens": token_usage.get('completion_tokens', 0),
        "total_tokens": token_usage.get('total_tokens', 0),
    }