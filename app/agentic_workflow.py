#app/agentic_workflow.py

from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState

from app.retriever import create_retriever
from config import settings
from app.logging_config import logger

# --- Helper function to format message content ---
def format_messages_for_llm(messages: list) -> str:
    """Strips metadata and formats messages as a simple string for the LLM."""
    formatted = []
    for msg in messages:
        role = "User" if msg.type == "human" else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)

# ... (GraphState and retrieve_documents are unchanged) ...
class GraphState(MessagesState):
    documents: List[Document]
    summary: str = ""
    turn_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

def retrieve_documents(state: GraphState) -> dict:
    logger.info("---NODE: RETRIEVE DOCUMENTS---")
    last_message = state['messages'][-1]
    question = last_message.content
    retriever, client = create_retriever()
    documents = retriever.invoke(question)
    client.close()
    return {"documents": documents}

# --- Refactor the summarization node for a complete initial summary ---
def summarize_history(state: GraphState) -> dict:
    """
    Node to create an append-only summary of the chat history.
    Handles the initial summary differently from subsequent updates.
    """
    logger.info("---NODE: SUMMARIZE HISTORY---")
    messages = state["messages"]
    summary = state.get("summary", "")
    turn_count = state.get("turn_count", 0) + 1
    
    # Determine which messages to summarize based on the turn count
    new_messages_to_summarize = []
    
    # At turn 3, summarize the first two turns (4 messages)
    if turn_count == 3:
        logger.info("---CREATING INITIAL SUMMARY---")
        # All messages except the current user query
        new_messages_to_summarize = messages[:-1]
    # For all subsequent turns, summarize only the last completed turn
    elif turn_count > 3:
        logger.info("---UPDATING SUMMARY---")
        new_messages_to_summarize = messages[-3:-1]

    if new_messages_to_summarize:
        summarization_prompt = ChatPromptTemplate.from_template(
            """Concisely summarize the following conversation:
            {new_messages}
            
            Concise Summary:"""
        )
        llm = ChatOpenAI(
            model_name="gpt-4.1-nano",
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.7
        )
        summarization_chain = summarization_prompt | llm
        
        formatted_new_messages = format_messages_for_llm(new_messages_to_summarize)

        summary_response = summarization_chain.invoke({
            "new_messages": formatted_new_messages
        })
        turn_summary = summary_response.content
        token_usage = summary_response.response_metadata.get('token_usage', {})

        # Append the new summary to the existing one
        new_summary = f"{summary}\n- {turn_summary}"
        
        return {
            "summary": new_summary.strip(), 
            "turn_count": turn_count,
            "prompt_tokens": token_usage.get('prompt_tokens', 0),
            "completion_tokens": token_usage.get('completion_tokens', 0),
            "total_tokens": token_usage.get('total_tokens', 0),
        }

    return {"turn_count": turn_count}


# --- Refactor the generation node ---
def generate_answer(state: GraphState) -> dict:
    """
    Node to generate an answer, now with corrected and simplified history.
    """
    logger.info("---NODE: GENERATE ANSWER---")
    question = state['messages'][-1].content
    documents = state["documents"]
    summary = state.get("summary", "")
    prompt_tokens = state.get("prompt_tokens", 0)
    completion_tokens = state.get("completion_tokens", 0)
    total_tokens = state.get("total_tokens", 0)
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following summary, context, and chat history. Don't make up information.
        
        Conversation Summary:
        {summary}

        Retrieved Context:
        {context}
        
        Recent Chat History (for immediate context):
        {chat_history}

        Question: {question}"""
    )
    llm = ChatOpenAI(
        model_name=settings.OPENAI_MODEL_NAME,
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=0.7
    )
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = prompt | llm
    
    # Get the last COMPLETED turn (human/ai pair) for recent history
    # Check if there are enough messages to form a complete previous turn
    recent_history_messages = state['messages'][-3:-1] if len(state['messages']) > 2 else []
    formatted_recent_history = format_messages_for_llm(recent_history_messages)
    
    generation = rag_chain.invoke({
        "question": question, 
        "context": documents,
        "summary": summary,
        "chat_history": formatted_recent_history 
    })
    token_usage = generation.response_metadata.get('token_usage', {})
    
    prompt_tokens += token_usage.get('prompt_tokens', 0)
    completion_tokens += token_usage.get('completion_tokens', 0)
    total_tokens += token_usage.get('total_tokens', 0)
    
    return {
        "messages": [generation],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

# --- Add the new Grounding and Safety Node ---
def grounding_and_safety_check(state: GraphState) -> dict:
    """
    Node to perform a grounding check and add citations.
    """
    logger.info("---NODE: GROUNDING & SAFETY CHECK---")
    question = state['messages'][-2].content # The last human message
    answer = state['messages'][-1].content # The last AI message (the answer)
    documents = state["documents"]
    
    # Accumulate token counts from the previous steps
    prompt_tokens = state.get("prompt_tokens", 0)
    completion_tokens = state.get("completion_tokens", 0)
    total_tokens = state.get("total_tokens", 0)

    # Prompt for the safety agent
    grounding_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant that acts as a final quality check.
        Your task is to review a generated answer based on a given question and a set of source documents.

        Here is the original question:
        {question}

        Here are the source documents:
        {context}

        Here is the generated answer:
        {answer}

        Please perform the following two tasks:
        1. Fact Check:
            - Verify that every claim in the generated answer is supported by the information in the source documents. 
            - If a claim is not supported, rewrite the answer to be accurate. Don't add citations if claim is not supported by the information in source documents.
            - If the answer is completely unsupported, respond with "I'm sorry, but I cannot answer that question based on the provided documents."
        2. Add Citations:
            - Only for claims in the answer that is supported by a document, add a citation in IEEE format i.e. in-text citations.
            - Refer to the source with a number in a square bracket, e.g. [1], that will then correspond to the full citation in your reference list.
            - List all references numerically in the order they've been cited within the answer, and include the bracketed number at the beginning of each reference.
            - Refer the below example for how to format citations in the answer. Stay consistent with this style throughout the conversation."

        Example of a citation:
        ```
        The sky is blue due to Rayleigh scattering [1]. The blue color is caused by the scattering of sunlight by the atmosphere [2].

        References:
        [1] Source Name or File Name
        [2] Another Source Name or File Name
        ```

        Return only the final, verified, and cited answer. Provide citations only if the claim is supported by the retrieved information.
        """
    )

    llm = ChatOpenAI(
        model_name="gpt-4.1-nano", # Use a fast and capable model for checking
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=0.7
    )

    def format_docs_for_citation(docs: List[Document]) -> str:
        """Formats docs with numbered sources for citation."""
        formatted = []
        for i, doc in enumerate(docs):
            source_id = f"{i+1}: {doc.metadata.get('file_name', 'N/A')}"
            formatted.append(f"[{source_id}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    grounding_chain = grounding_prompt | llm
    
    # Invoke the chain with the necessary information
    response = grounding_chain.invoke({
        "question": question,
        "context": format_docs_for_citation(documents),
        "answer": answer
    })
    
    # Update the last message in the history with the verified answer
    final_answer = response.content
    state['messages'][-1].content = final_answer

    token_usage = response.response_metadata.get('token_usage', {})
    
    # Add the new token usage to the existing counts
    prompt_tokens += token_usage.get('prompt_tokens', 0)
    completion_tokens += token_usage.get('completion_tokens', 0)
    total_tokens += token_usage.get('total_tokens', 0)

    # Return the updated state
    return {
        "messages": state['messages'],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }