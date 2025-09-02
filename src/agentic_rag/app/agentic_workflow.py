# src/agentic_rag/app/agentic_workflow.py

from typing import List, Literal
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.tools.tavily_search import TavilySearchResults

from agentic_rag.app.retriever import create_retriever
from agentic_rag.app.llm_provider import get_llm
from agentic_rag.logging_config import logger
from agentic_rag.config import settings

# --- Pydantic Models for Structured Output ---


class QueryClassification(BaseModel):
    """The classification of the user's query."""

    classification: Literal["simple", "complex"] = Field(
        ...,
        description="The classification of the user's query, either 'simple' or 'complex'.",
    )


class GroundingCheck(BaseModel):
    """The result of a grounding and safety check."""

    is_grounded: bool = Field(
        ...,
        description="Whether the answer is fully supported by the provided source documents.",
    )
    revised_answer: str = Field(
        ...,
        description="The revised, fact-checked, and cited answer. If not grounded, this should be a message indicating failure.",
    )


# --- Helper function to format message content ---
def format_messages_for_llm(messages: list) -> str:
    """Strips metadata and formats messages as a simple string for the LLM."""
    formatted = []
    for msg in messages:
        role = "User" if msg.type == "human" else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)


# --- Updated GraphState with self-correction fields ---
class GraphState(MessagesState):
    documents: List[Document]
    initial_documents: List[Document]  # To store the initially retrieved docs
    summary: str = ""
    turn_count: int = 0
    transformed_query: str | None = None
    hyde_document: str | None = None
    is_complex_query: bool = False
    retrieval_success: bool = False
    needs_reranking: bool = False
    grounding_success: bool = True
    is_web_search: bool = False
    retrieval_retries: int = 0
    grounding_retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# --- Agentic Nodes ---


def classify_query(state: GraphState) -> dict:
    """
    Classifies the user's query using structured output to determine if it is
    a simple standalone question or a complex one that requires conversation history.
    """
    logger.info("---NODE: CLASSIFY QUERY---")
    last_message = state["messages"][-1].content
    conversation_history = format_messages_for_llm(state["messages"][:-1])

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
    # Use structured output
    llm = get_llm(fast_model=True).with_structured_output(QueryClassification)
    chain = prompt | llm

    result = chain.invoke({"history": conversation_history, "query": last_message})

    is_complex = result.classification == "complex"
    logger.info(f"Query classified as: {'complex' if is_complex else 'simple'}")

    return {"is_complex_query": is_complex}


def transform_query(state: GraphState) -> dict:
    """
    Rewrites the user's query into a more precise, standalone question
    that is optimized for vector retrieval.
    """
    logger.info("---NODE: TRANSFORM QUERY---")
    conversation_history = format_messages_for_llm(state["messages"])

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


def generate_hyde_document(state: GraphState) -> dict:
    """Generates a hypothetical answer to be used for retrieval."""
    logger.info("---NODE: GENERATE HYDE DOCUMENT---")
    query = state.get("transformed_query") or state["messages"][-1].content
    prompt = ChatPromptTemplate.from_template(
        "Generate a concise, hypothetical answer to the following question: {question}"
    )
    llm = get_llm(fast_model=True)
    chain = prompt | llm
    result = chain.invoke({"question": query})
    return {"hyde_document": result.content}


def web_search(state: GraphState) -> dict:
    """Performs a web search using the Tavily API."""
    logger.info("---NODE: WEB SEARCH---")
    query = state.get("transformed_query") or state["messages"][-1].content
    tool = TavilySearchResults(max_results=3)
    documents = tool.invoke(query)
    # The tool returns a list of dicts, we need to convert them to Document objects
    doc_objects = [
        Document(page_content=doc["content"], metadata={"source": doc["url"]})
        for doc in documents
    ]
    return {
        "documents": doc_objects,
        "retrieval_success": bool(doc_objects),
        "is_web_search": True,
    }


def retrieve_documents(state: GraphState) -> dict:
    """
    Retrieves documents using either the query or a HyDE document for embedding.
    """
    logger.info("---NODE: RETRIEVE DOCUMENTS---")
    query = (
        state.get("hyde_document")
        or state.get("transformed_query")
        or state["messages"][-1].content
    )

    retriever, client = create_retriever()
    documents = retriever.invoke(query)
    client.close()

    needs_reranking = bool(documents)
    return {
        "documents": documents,
        "initial_documents": documents,
        "needs_reranking": needs_reranking,
        "is_web_search": False,
        "hyde_document": None,
    }


def grade_and_rerank_documents(state: GraphState) -> dict:
    """
    Re-ranks retrieved documents based on their relevance to the query using a Cross-Encoder.
    """
    logger.info("---NODE: RE-RANK DOCUMENTS---")
    query = state.get("transformed_query") or state["messages"][-1].content
    documents = state["documents"]

    cross_encoder = HuggingFaceCrossEncoder(
        model_name=settings.CROSS_ENCODER_MODEL_SMALL
    )
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.score(pairs)
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in scored_docs[:3]]

    retrieval_success = bool(reranked_docs)
    logger.info(
        f"Re-ranking complete. Selected top {len(reranked_docs)} relevant documents."
    )

    return {"documents": reranked_docs, "retrieval_success": retrieval_success}


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

        summary_response = summarization_chain.invoke(
            {"new_messages": formatted_new_messages}
        )
        turn_summary = summary_response.content
        token_usage = summary_response.response_metadata.get("token_usage", {})

        # Append the new summary to the existing one
        new_summary = f"{summary}\n- {turn_summary}"

        return {
            "summary": new_summary.strip(),
            "turn_count": turn_count,
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
        }

    return {"turn_count": turn_count}


def generate_answer(state: GraphState) -> dict:
    """
    Node to generate an answer, now with corrected and simplified history.
    """
    logger.info("---NODE: GENERATE ANSWER---")
    question = state["messages"][-1].content
    documents = state["documents"]
    summary = state.get("summary", "")
    prompt_tokens = state.get("prompt_tokens", 0)
    completion_tokens = state.get("completion_tokens", 0)
    total_tokens = state.get("total_tokens", 0)

    prompt_template = """Answer the question based only on the following summary, context, and chat history. Don't make up information.
        
        Conversation Summary:
        {summary}

        Retrieved Context:
        {context}
        
        Recent Chat History (for immediate context):
        {chat_history}

        Question: {question}"""

    # Add a stricter instruction for grounding retries
    if state.get("grounding_retries", 0) > 0:
        logger.info("Applying stricter prompt for re-generation.")
        prompt_template += "\n\nCRITICAL: Ensure every single claim in your answer is directly supported by the provided context. Do not infer or add outside information."

    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = get_llm(fast_model=True)
    llm.temperature = 0.1

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = prompt | llm

    # Get the last COMPLETED turn (human/ai pair) for recent history
    # Check if there are enough messages to form a complete previous turn
    recent_history_messages = (
        state["messages"][-3:-1] if len(state["messages"]) > 2 else []
    )
    formatted_recent_history = format_messages_for_llm(recent_history_messages)

    generation = rag_chain.invoke(
        {
            "question": question,
            "context": format_docs(documents),
            "summary": summary,
            "chat_history": formatted_recent_history,
        }
    )
    token_usage = generation.response_metadata.get("token_usage", {})

    prompt_tokens += token_usage.get("prompt_tokens", 0)
    completion_tokens += token_usage.get("completion_tokens", 0)
    total_tokens += token_usage.get("total_tokens", 0)

    return {
        "messages": [generation],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def grounding_and_safety_check(state: GraphState) -> dict:
    """
    Performs a grounding check using structured output and revises the answer.
    Sets a flag based on whether the answer is grounded in the context.
    """
    logger.info("---NODE: GROUNDING & SAFETY CHECK---")
    question = state["messages"][-2].content
    answer = state["messages"][-1].content
    documents = state["documents"]

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
        1. Verify that every claim in the generated answer is supported by the information in the source documents.
        2. If the answer is fully supported, revise it to include citations in IEEE format (e.g., [1], [2]). List the sources in a 'References' section at the end.
        3. If the answer is not supported or contains hallucinations, set 'is_grounded' to false and provide a revised answer stating that you cannot answer based on the documents.

        Return the result in the requested structured format.
        """
    )

    llm = get_llm(fast_model=False).with_structured_output(GroundingCheck)

    def format_docs_for_citation(docs: List[Document]) -> str:
        """Formats docs with numbered sources for citation."""
        formatted = []
        for i, doc in enumerate(docs):
            source_id = f"Source [{i + 1}]: {doc.metadata.get('file_name', 'N/A')}"
            formatted.append(f"{source_id}\n{doc.page_content}")
        return "\n\n".join(formatted)

    grounding_chain = grounding_prompt | llm

    response = grounding_chain.invoke(
        {
            "question": question,
            "context": format_docs_for_citation(documents),
            "answer": answer,
        }
    )

    logger.info(f"Grounding check complete. Is grounded: {response.is_grounded}")

    # Get the ID of the last AI message to replace it instead of appending
    # This should always exist since grounding check follows generate_answer
    if state["messages"] and state["messages"][-1].type == "ai":
        last_ai_message = state["messages"][-1]
        revised_message = AIMessage(
            content=response.revised_answer, 
            id=last_ai_message.id
        )
    else:
        # Fallback - create new message (this shouldn't happen in current workflow)
        revised_message = AIMessage(content=response.revised_answer)

    return {
        "messages": [revised_message],
        "grounding_success": response.is_grounded,
    }


def web_search_safety_check(state: GraphState) -> dict:
    """A simplified safety check for web search results that adds citations."""
    logger.info("---NODE: WEB SEARCH SAFETY CHECK---")
    answer = state["messages"][-1].content
    documents = state["documents"]

    cited_answer = f"{answer}\n\n**Sources:**\n"
    for i, doc in enumerate(documents):
        source_url = doc.metadata.get("source", "N/A")
        cited_answer += f"[{i + 1}] {source_url}\n"

    # Get the ID of the last AI message to replace it instead of appending
    # This should always exist since web search safety check follows generate_answer
    if state["messages"] and state["messages"][-1].type == "ai":
        last_ai_message = state["messages"][-1]
        revised_message = AIMessage(
            content=cited_answer, 
            id=last_ai_message.id
        )
    else:
        # Fallback - create new message (this shouldn't happen in current workflow)
        revised_message = AIMessage(content=cited_answer)

    return {"messages": [revised_message]}


def handle_retrieval_failure(state: GraphState) -> dict:
    logger.info("---NODE: HANDLE RETRIEVAL FAILURE---")
    
    # Get the ID of the last AI message to replace it instead of appending
    # This should always exist in the current workflow, but add defensive check
    if state["messages"] and state["messages"][-1].type == "ai":
        last_ai_message = state["messages"][-1]
        failure_message = AIMessage(
            content="I'm sorry, but I couldn't find any information to answer your question, even after trying multiple strategies.",
            id=last_ai_message.id
        )
    else:
        # Fallback - create new message (this shouldn't happen in current workflow)
        failure_message = AIMessage(
            content="I'm sorry, but I couldn't find any information to answer your question, even after trying multiple strategies."
        )
    
    return {
        "messages": [failure_message]
    }


def handle_grounding_failure(state: GraphState) -> dict:
    logger.info("---NODE: HANDLE GROUNDING FAILURE---")
    
    # Get the ID of the last AI message to replace it instead of appending
    # This should always exist in the current workflow, but add defensive check
    if state["messages"] and state["messages"][-1].type == "ai":
        last_ai_message = state["messages"][-1]
        failure_message = AIMessage(
            content="I found some information, but I could not construct a factually grounded answer. Please try rephrasing your question.",
            id=last_ai_message.id
        )
    else:
        # Fallback - create new message (this shouldn't happen in current workflow)
        failure_message = AIMessage(
            content="I found some information, but I could not construct a factually grounded answer. Please try rephrasing your question."
        )
    
    return {
        "messages": [failure_message]
    }


def increment_retrieval_retry_counter(state: GraphState) -> dict:
    logger.info("---NODE: INCREMENT RETRIEVAL RETRY COUNTER---")
    retries = state.get("retrieval_retries", 0) + 1
    return {"retrieval_retries": retries}


def increment_grounding_retry_counter(state: GraphState) -> dict:
    logger.info("---NODE: INCREMENT GROUNDING RETRY COUNTER---")
    retries = state.get("grounding_retries", 0) + 1
    return {"grounding_retries": retries}


def smart_retrieval_and_rerank(state: GraphState) -> dict:
    """
    A more powerful retrieval and re-ranking step for grounding correction.
    Uses the larger cross-encoder model.
    """
    logger.info("---NODE: SMART RETRIEVAL & RE-RANK (GROUNDING CORRECTION)---")
    query = state.get("transformed_query") or state["messages"][-1].content
    # Use the initial, unfiltered documents for re-ranking
    documents = state["initial_documents"]

    cross_encoder = HuggingFaceCrossEncoder(
        model_name=settings.CROSS_ENCODER_MODEL_LARGE
    )
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.score(pairs)
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    # Keep a slightly larger set of top documents for better context
    reranked_docs = [doc for doc, score in scored_docs[:4]]

    logger.info(
        f"Smart re-ranking complete. Selected top {len(reranked_docs)} documents."
    )

    return {"documents": reranked_docs}


def hybrid_context_retrieval(state: GraphState) -> dict:
    """
    Combines the best internal documents with fresh web search results.
    """
    logger.info("---NODE: HYBRID CONTEXT RETRIEVAL (GROUNDING CORRECTION)---")
    internal_documents = state["documents"]

    # Perform a web search
    web_search_state = web_search(state)
    web_documents = web_search_state.get("documents", [])

    # Combine and de-duplicate
    combined_docs = internal_documents + web_documents

    # Optional: A final re-ranking could be done here on the combined set

    logger.info(
        f"Combined {len(internal_documents)} internal docs with {len(web_documents)} web docs."
    )

    return {"documents": combined_docs, "is_web_search": True}


# --- Routers ---


def route_for_retrieval(state: GraphState) -> str:
    logger.info("---ROUTER: ROUTE FOR RETRIEVAL---")
    if state.get("is_complex_query"):
        return "transform_query"
    else:
        return "retrieve"


def route_after_retrieval(state: GraphState) -> str:
    logger.info("---ROUTER: ROUTE AFTER RETRIEVAL---")
    if state.get("needs_reranking"):
        return "rerank_documents"
    else:
        return "enter_retrieval_correction"


def route_after_reranking(state: GraphState) -> str:
    logger.info("---ROUTER: ROUTE AFTER RE-RANKING---")
    if state.get("retrieval_success"):
        return "summarize"
    else:
        return "enter_retrieval_correction"


def route_after_generation(state: GraphState) -> str:
    """
    Router that decides which safety check to use based on the source of the documents.
    """
    logger.info("---ROUTER: ROUTE AFTER GENERATION---")
    if state.get("is_web_search"):
        logger.info("Routing to: web_search_safety_check")
        return "web_search_safety_check"
    else:
        logger.info("Routing to: safety_check")
        return "safety_check"


def route_after_safety_check(state: GraphState) -> str:
    logger.info("---ROUTER: ROUTE AFTER SAFETY CHECK---")
    if state.get("grounding_success"):
        return "END"
    else:
        return "enter_grounding_correction"


def route_retrieval_correction(state: GraphState) -> str:
    """
    Routes to the next retrieval correction strategy based on the retry count.
    """
    logger.info("---ROUTER: ROUTE RETRIEVAL CORRECTION STRATEGY---")
    retries = state.get("retrieval_retries", 0)

    if retries == 1:
        logger.info("Correction Strategy 1: Query Transformation")
        return "transform_query"
    elif retries == 2:
        logger.info("Correction Strategy 2: HyDE")
        return "generate_hyde_document"
    elif retries == 3:
        logger.info("Correction Strategy 3: Web Search")
        return "web_search"
    else:
        logger.warning("Max retrieval retries exceeded. Routing to failure handler.")
        return "handle_retrieval_failure"


def route_grounding_correction(state: GraphState) -> str:
    """
    Routes to the next grounding correction strategy with an escalating approach.
    """
    logger.info("---ROUTER: ROUTE GROUNDING CORRECTION STRATEGY---")
    retries = state.get("grounding_retries", 0)

    if retries == 1:
        logger.info("Grounding Correction Strategy 1: Smart Retrieval & Re-Rank")
        return "smart_retrieval"
    elif retries == 2:
        logger.info("Grounding Correction Strategy 2: Hybrid Context (Internal + Web)")
        return "hybrid_context"
    else:
        logger.warning("Max grounding retries exceeded. Routing to failure handler.")
        return "handle_grounding_failure"
