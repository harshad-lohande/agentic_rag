# app/agentic_workflow.py

from typing import List, Literal
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState, END
from pydantic import BaseModel, Field

from app.retriever import create_retriever
from app.llm_provider import get_llm
from app.logging_config import logger

# --- Pydantic Models for Structured Output ---

class QueryClassification(BaseModel):
    """The classification of the user's query."""
    classification: Literal["simple", "complex"] = Field(
        ..., description="The classification of the user's query, either 'simple' or 'complex'."
    )

class VerifiedClaim(BaseModel):
    """A single claim or statement made in the generated answer, with verification."""
    statement: str = Field(..., description="A single claim or statement made in the generated answer.")
    is_supported: bool = Field(..., description="Whether the statement is directly supported by the provided source documents.")
    supporting_sources: List[int] = Field(..., description="A list of source numbers (e.g., [1], [2]) that support the claim.")

class GroundedAnswer(BaseModel):
    """The final, grounded answer composed of a list of verified claims."""
    verified_claims: List[VerifiedClaim]

# --- Helper function to format message content ---
def format_messages_for_llm(messages: list) -> str:
    """Strips metadata and formats messages as a simple string for the LLM."""
    formatted = []
    for msg in messages:
        role = "User" if msg.type == "human" else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)

# --- GraphState with decision-making flags ---
class GraphState(MessagesState):
    documents: List[Document]
    summary: str = ""
    turn_count: int = 0
    # The transformed query for retrieval
    transformed_query: str | None = None
    # Flag to indicate if the query is complex
    is_complex_query: bool = False
    # Flag to indicate if retrieval was successful
    retrieval_success: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# --- Node: Classify Query with Structured Output ---
def classify_query(state: GraphState) -> dict:
    """
    Classifies the user's query to determine if it's a simple standalone question
    or a complex one that requires conversation history.
    """
    logger.info("---NODE: CLASSIFY QUERY---")
    last_message = state['messages'][-1].content
    conversation_history = format_messages_for_llm(state['messages'][:-1])

    prompt = ChatPromptTemplate.from_template(
        """You are an expert at analyzing conversations.
        Your task is to determine if the user's latest query is a simple, standalone question or 
        if it's a complex question that depends on the previous conversation history.

        Respond with "simple" or "complex".

        Conversation History:
        {history}

        User Query: {query}
        """
    )
    llm = get_llm(fast_model=True).with_structured_output(QueryClassification)
    chain = prompt | llm

    result = chain.invoke({"history": conversation_history, "query": last_message})
    
    is_complex = result.classification == "complex"
    logger.info(f"Query classified as: {'complex' if is_complex else 'simple'}")

    return {"is_complex_query": is_complex}


# --- Node: Transform Query ---
def transform_query(state: GraphState) -> dict:
    """
    Rewrites the user's query into a more precise, standalone question
    that is optimized for vector retrieval.
    """
    logger.info("---NODE: TRANSFORM QUERY---")
    last_message = state['messages'][-1].content
    conversation_history = format_messages_for_llm(state['messages'])

    prompt = ChatPromptTemplate.from_template(
        """You are an expert at rewriting conversational queries into standalone, optimized search queries.

        Based on the conversation history,
        rewrite the user's latest query into a clear, concise, and self-contained question 
        that can be used for a vector search.

        Conversation History:
        {history}

        Rewritten Query:
        """
    )
    llm = get_llm(fast_model=True)
    chain = prompt | llm

    result = chain.invoke({"history": conversation_history})
    logger.info(f"Transformed query: {result.content}")

    return {"transformed_query": result.content}


# --- Node: Retrieve Documents ---
def retrieve_documents(state: GraphState) -> dict:
    """
    Retrieves documents based on the transformed query, or the original query
    if no transformation was needed.
    """
    logger.info("---NODE: RETRIEVE DOCUMENTS---")
    # Use the transformed query if it exists, otherwise use the original
    query = state.get("transformed_query") or state['messages'][-1].content
    
    retriever, client = create_retriever()
    documents = retriever.invoke(query)
    client.close()
    
    # Set the retrieval_success flag
    retrieval_success = bool(documents)
    logger.info(f"Retrieval success: {retrieval_success}")

    return {"documents": documents, "retrieval_success": retrieval_success}


# --- Node: Route for Retrieval (Router) ---
def route_for_retrieval(state: GraphState) -> str:
    """
    The first router in our graph. Decides whether to transform the query
    or retrieve documents directly.
    """
    logger.info("---ROUTER: ROUTE FOR RETRIEVAL---")
    if state.get("is_complex_query"):
        logger.info("Routing to: transform_query")
        return "transform_query"
    else:
        logger.info("Routing to: retrieve")
        return "retrieve"

# --- Node: Route for Generation (Router) ---
def route_for_generation(state: GraphState) -> str:
    """
    The second router in our graph. Decides whether to generate an answer
    or handle a retrieval failure.
    """
    logger.info("---ROUTER: ROUTE FOR GENERATION---")
    if state.get("retrieval_success"):
        logger.info("Routing to: summarize")
        return "summarize"
    else:
        logger.info("Routing to: retrieval_failure")
        # In a more advanced setup, this could route to a web search tool
        return "retrieval_failure"

# --- Node: Handle Retrieval Failure ---
def handle_retrieval_failure(state: GraphState) -> dict:
    """
    A simple node to handle cases where no documents were retrieved.
    """
    logger.info("---NODE: HANDLE RETRIEVAL FAILURE---")
    return {
        "messages": [
            "I'm sorry, but I couldn't find any information in the provided documents to answer your question."
        ]
    }

# --- Node: Summarize conversation history ---
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
        llm = get_llm(fast_model=True)
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


# --- Node: Generate answer based on summary, context, and chat history ---
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

    llm = get_llm(fast_model=True)
    
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

# --- Node: Grounding and Safety Node with Structured Output ---
def grounding_and_safety_check(state: GraphState) -> dict:
    """
    Performs a grounding check using structured output and reconstructs the
    final answer with reliable citations.
    """
    logger.info("---NODE: GROUNDING & SAFETY CHECK---")
    question = state['messages'][-2].content
    answer = state['messages'][-1].content
    documents = state["documents"]
    
    prompt_tokens = state.get("prompt_tokens", 0)
    completion_tokens = state.get("completion_tokens", 0)
    total_tokens = state.get("total_tokens", 0)

    grounding_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant that acts as a final quality check.
        Your task is to review a generated answer based on a set of source documents and a question.

        Here is the original question:
        {question}

        Here are the source documents, each with a source number:
        {context}

        Here is the generated answer that you need to verify:
        {answer}

        Please perform the following tasks:
        1. Break down the generated answer into individual claims/statements.
        2. For each statement, verify if it is supported by the provided source documents.
        3. For each supported statement, identify the source numbers of the documents that support it.
        4. If the answer is completely unsupported, respond with a single claim stating that you cannot answer the question.
        
        Return a list of verified claims in the requested structured format.
        """
    )

    llm = get_llm(fast_model=False).with_structured_output(GroundedAnswer)

    def format_docs_for_citation(docs: List[Document]) -> str:
        """Formats docs with numbered sources for citation."""
        formatted = []
        for i, doc in enumerate(docs):
            source_id = f"Source [{i+1}]: {doc.metadata.get('file_name', 'N/A')}"
            formatted.append(f"{source_id}\n{doc.page_content}")
        return "\n\n".join(formatted)

    grounding_chain = grounding_prompt | llm
    
    response = grounding_chain.invoke({
        "question": question,
        "context": format_docs_for_citation(documents),
        "answer": answer
    })
    
    # --- Programmatically reconstruct the final answer ---
    final_answer_parts = []
    cited_sources = set()
    for claim in response.verified_claims:
        if claim.is_supported and claim.supporting_sources:
            citations = [f"[{source_num}]" for source_num in claim.supporting_sources]
            cited_sources.update(claim.supporting_sources)
            final_answer_parts.append(f"{claim.statement} {''.join(citations)}")
        else:
            final_answer_parts.append(claim.statement)
    
    final_answer = " ".join(final_answer_parts)
    
    # Add the references section
    if cited_sources:
        final_answer += "\n\n**References:**\n"
        for source_num in sorted(list(cited_sources)):
            file_name = documents[source_num - 1].metadata.get('file_name', 'N/A')
            final_answer += f"[{source_num}] {file_name}\n"

    state['messages'][-1].content = final_answer

    # Token usage is not available in the structured output response, so we'll skip updating it for this node.
    
    return {
        "messages": state['messages'],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }