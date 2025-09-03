# src/agentic_rag/app/agentic_workflow.py

import hashlib
from typing import List, Literal, Dict, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage  # NEW

from agentic_rag.app.retriever import create_retriever
from agentic_rag.app.llm_provider import get_llm
from agentic_rag.app.message_utils import (
    _msg_type,
    get_message_content,
    get_last_message_content,
    get_last_human_message_content,
    get_last_ai_message_content,
    get_last_completed_turn_messages,
)
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
        msg_type = _msg_type(msg)
        role = "User" if msg_type == "human" else "Assistant"
        content = get_message_content(msg)
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


# --- Token Usage Helper Functions ---
def extract_token_usage_from_response(response) -> Dict[str, int]:
    """
    Extracts token usage from an LLM response.
    Handles both regular responses and structured output responses.
    Returns a dictionary with prompt_tokens, completion_tokens, and total_tokens.
    """
    token_usage = {}
    
    # For structured output, the response might be wrapped differently
    if hasattr(response, 'response_metadata'):
        token_usage = response.response_metadata.get("token_usage", {})
    elif hasattr(response, '__pydantic_extra__') and 'response_metadata' in getattr(response, '__pydantic_extra__', {}):
        # Some structured outputs store metadata in __pydantic_extra__
        token_usage = response.__pydantic_extra__.get('response_metadata', {}).get("token_usage", {})
    elif hasattr(response, '_response_metadata'):
        # Alternative attribute name
        token_usage = response._response_metadata.get("token_usage", {})
    
    return {
        "prompt_tokens": token_usage.get("prompt_tokens", 0),
        "completion_tokens": token_usage.get("completion_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
    }


def accumulate_token_usage(state: dict, new_tokens: Dict[str, int]) -> Dict[str, int]:
    """
    Accumulates token usage from state and new tokens.
    Returns the updated token counts.
    """
    current_prompt = state.get("prompt_tokens", 0)
    current_completion = state.get("completion_tokens", 0)
    current_total = state.get("total_tokens", 0)
    
    return {
        "prompt_tokens": current_prompt + new_tokens.get("prompt_tokens", 0),
        "completion_tokens": current_completion + new_tokens.get("completion_tokens", 0),
        "total_tokens": current_total + new_tokens.get("total_tokens", 0),
    }


# --- Helper functions for grounding correction improvements ---


def get_document_dedup_key(doc: Document) -> str:
    """
    Generate a stable de-duplication key for a document.
    Uses metadata in order of preference: source, file_name, path, id, or hash of content.
    """
    metadata = doc.metadata or {}

    # Try metadata fields in order of preference
    for key in ["source", "file_name", "path", "id"]:
        if key in metadata and metadata[key]:
            return str(metadata[key])

    # Fallback to hash of page_content
    content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
    return f"content_hash_{content_hash}"


def deduplicate_documents(documents: List[Document]) -> List[Document]:
    """
    Remove duplicate documents based on deduplication key, keeping first occurrence.
    Returns deduplicated list and logs counts.
    """
    if not documents:
        return documents

    seen_keys = set()
    deduplicated = []

    for doc in documents:
        key = get_document_dedup_key(doc)
        if key not in seen_keys:
            seen_keys.add(key)
            deduplicated.append(doc)

    if len(deduplicated) != len(documents):
        logger.info(
            f"Document deduplication: {len(documents)} -> {len(deduplicated)} documents"
        )

    return deduplicated


def apply_reciprocal_rank_fusion(
    doc_lists: List[List[Document]], k: int = 60
) -> List[Document]:
    """
    Apply Reciprocal Rank Fusion (RRF) to merge multiple document lists.
    Score formula: score(doc) = sum(1.0 / (k + rank_i)) across all lists where doc appears.
    """
    doc_scores: Dict[str, Tuple[Document, float]] = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            dedup_key = get_document_dedup_key(doc)
            score = 1.0 / (k + rank)

            if dedup_key in doc_scores:
                # Add to existing score, keep first occurrence of document
                doc_scores[dedup_key] = (
                    doc_scores[dedup_key][0],
                    doc_scores[dedup_key][1] + score,
                )
            else:
                doc_scores[dedup_key] = (doc, score)

    # Sort by RRF score in descending order
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
    fused_docs = [doc for doc, score in sorted_docs]

    logger.info(
        f"RRF fusion applied to {len(doc_lists)} lists, resulting in {len(fused_docs)} unique documents"
    )
    return fused_docs


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts using tokenized sets.
    Tokenizes by splitting on whitespace and converts to lowercase.
    """
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    if not union:
        return 0.0

    return len(intersection) / len(union)


def apply_diversity_filter(
    documents: List[Document], top_k: int = 4, similarity_threshold: float = 0.85
) -> List[Document]:
    """
    Apply diversity filter to remove documents that are too similar to previously selected ones.
    Uses Jaccard similarity with the specified threshold.
    """
    if not documents or top_k <= 0:
        return documents[:top_k]

    selected = []
    filtered_count = 0

    for doc in documents:
        if len(selected) >= top_k:
            break

        # Check similarity with already selected documents
        is_too_similar = False
        for selected_doc in selected:
            similarity = jaccard_similarity(doc.page_content, selected_doc.page_content)
            if similarity >= similarity_threshold:
                is_too_similar = True
                filtered_count += 1
                break

        if not is_too_similar:
            selected.append(doc)

    if filtered_count > 0:
        logger.info(
            f"Diversity filter: {filtered_count} documents filtered for redundancy"
        )

    return selected


def inline_transform_query(original_query: str) -> str:
    """
    Inline transformation of query with drift-avoidance guard.
    Uses the same pattern as transform_query but with explicit constraints.
    """
    prompt = ChatPromptTemplate.from_template(
        """You are an expert at rewriting conversational queries into standalone, optimized search queries.

        Rewrite the user's query into a clear, concise, and self-contained question 
        that can be used for a vector search.

        IMPORTANT: Preserve key entities, numbers, dates, and constraints from the user's question. 
        Do not introduce new facts or assumptions.

        Original Query: {query}

        Rewritten Query:
        """
    )

    llm = get_llm(fast_model=True)
    llm.temperature = 0.1  # Keep temperature low for consistency
    chain = prompt | llm

    result = chain.invoke({"query": original_query})
    return result.content


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
    proposed_answer: str | None = None  # NEW: hold draft answer before finalization


# --- Agentic Nodes ---


def classify_query(state: GraphState) -> dict:
    """
    Classifies the user's query using structured output to determine if it is
    a simple standalone question or a complex one that requires conversation history.
    """
    logger.info("---NODE: CLASSIFY QUERY---")
    last_message = get_last_message_content(state["messages"])
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

    # Extract and accumulate token usage
    new_tokens = extract_token_usage_from_response(result)
    accumulated_tokens = accumulate_token_usage(state, new_tokens)

    is_complex = result.classification == "complex"
    logger.info(f"Query classified as: {'complex' if is_complex else 'simple'}")

    return {
        "is_complex_query": is_complex,
        **accumulated_tokens,
    }


def transform_query(state: GraphState) -> dict:
    """
    Rewrite the latest user query to be fully self-contained.
    We include a small recent history window (last completed human–AI pair) to resolve pronouns and references
    with minimal token cost.
    """
    messages = state.get("messages", [])
    # Extract the latest user question
    question = get_last_human_message_content(messages)

    # Use the last completed turn (one human + following AI) as light context for coref
    # Fall back to empty if not available (e.g., first user turn)
    try:
        # get_last_completed_turn_messages should return [HumanMessage, AIMessage] for the previous completed turn
        recent_turn = get_last_completed_turn_messages(messages, k=1)
        recent_history = format_messages_for_llm(recent_turn)
    except Exception:
        recent_history = ""

    # Feature toggle: allow disabling this extra context if needed (defaults to True)
    use_recent_in_rewrite = bool(
        getattr(settings, "USE_RECENT_HISTORY_IN_REWRITE", True)
    )
    history_for_prompt = recent_history if use_recent_in_rewrite else ""

    # Build the prompt and call LLM
    prompt = ChatPromptTemplate.from_template(
        "You will rewrite the latest user question to be fully self-contained and unambiguous.\n"
        "Use the recent conversation to resolve pronouns and references if provided.\n\n"
        "Recent conversation (may be empty):\n{chat_history}\n\n"
        "Original question:\n{original}\n\n"
        "Rewritten question:"
    )
    chain = prompt | get_llm(fast_model=True)
    result = chain.invoke({"chat_history": history_for_prompt, "original": question})

    # Extract and accumulate token usage
    new_tokens = extract_token_usage_from_response(result)
    accumulated_tokens = accumulate_token_usage(state, new_tokens)

    rewritten = (
        result.content.strip() if hasattr(result, "content") else str(result).strip()
    )
    return {
        "transformed_query": rewritten,
        **accumulated_tokens,
    }


def generate_hyde_document(state: GraphState) -> dict:
    """Generates a hypothetical answer to be used for retrieval."""
    logger.info("---NODE: GENERATE HYDE DOCUMENT---")
    query = state.get("transformed_query") or get_last_message_content(
        state["messages"]
    )
    prompt = ChatPromptTemplate.from_template(
        "Generate a concise, hypothetical answer to the following question: {question}"
    )
    llm = get_llm(fast_model=True)
    chain = prompt | llm
    result = chain.invoke({"question": query})
    
    # Extract and accumulate token usage
    new_tokens = extract_token_usage_from_response(result)
    accumulated_tokens = accumulate_token_usage(state, new_tokens)
    
    return {
        "hyde_document": result.content,
        **accumulated_tokens,
    }


def web_search(state: GraphState) -> dict:
    """Performs a web search using the Tavily API."""
    logger.info("---NODE: WEB SEARCH---")
    query = state.get("transformed_query") or get_last_message_content(
        state["messages"]
    )
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
        or get_last_message_content(state["messages"])
    )

    retriever, client = create_retriever()
    documents = retriever.invoke(query)
    dedup_documents = deduplicate_documents(documents)
    client.close()

    needs_reranking = bool(documents)
    return {
        "documents": dedup_documents,
        "initial_documents": dedup_documents,
        "needs_reranking": needs_reranking,
        "is_web_search": False,
        "hyde_document": None,
    }


def grade_and_rerank_documents(state: GraphState) -> dict:
    """
    Re-ranks retrieved documents based on their relevance to the query using a Cross-Encoder.
    """
    logger.info("---NODE: RE-RANK DOCUMENTS---")
    query = state.get("transformed_query") or get_last_message_content(
        state["messages"]
    )
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
        # Use robust helper to get the last completed turn instead of fragile slicing
        new_messages_to_summarize = get_last_completed_turn_messages(messages)

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
    IMPORTANT: Do not append to messages here. Save only a proposed_answer.
    """
    logger.info("---NODE: GENERATE ANSWER---")
    question = get_last_human_message_content(state["messages"])
    documents = state["documents"]
    summary = state.get("summary", "")
    prompt_tokens = state.get("prompt_tokens", 0)
    completion_tokens = state.get("completion_tokens", 0)
    total_tokens = state.get("total_tokens", 0)

    prompt_template = """Answer the question based only on the following summary, context, and chat history. Don't make up information
        
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

    # Get the last COMPLETED turn (human/ai pair) for recent history using robust helper
    recent_history_messages = get_last_completed_turn_messages(state["messages"])
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

    # Only store the proposed answer here; do not touch messages
    return {
        "proposed_answer": generation.content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def grounding_and_safety_check(state: GraphState) -> dict:
    """
    Performs a grounding check using structured output and revises the answer.
    Appends a single assistant message ONLY if grounded; otherwise appends nothing
    (so that correction loops do not create extra messages).
    """
    logger.info("---NODE: GROUNDING & SAFETY CHECK---")
    question = get_last_human_message_content(state["messages"])
    # Prefer the proposed_answer (not yet appended to messages)
    answer = state.get("proposed_answer") or get_last_ai_message_content(
        state["messages"]
    )
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

    # Extract and accumulate token usage
    new_tokens = extract_token_usage_from_response(response)
    accumulated_tokens = accumulate_token_usage(state, new_tokens)

    logger.info(f"Grounding check complete. Is grounded: {response.is_grounded}")

    # Append the final assistant message ONLY when grounded
    if response.is_grounded:
        return {
            "messages": [AIMessage(content=response.revised_answer)],
            "grounding_success": True,
            "proposed_answer": None,
            **accumulated_tokens,
        }
    else:
        # Do not append anything; move into correction loop
        return {
            "grounding_success": False,
            **accumulated_tokens,
        }


def web_search_safety_check(state: GraphState) -> dict:
    """A simplified safety check for web search results that adds citations.
    Appends a single assistant message with citations."""
    logger.info("---NODE: WEB SEARCH SAFETY CHECK---")
    # Prefer the proposed_answer (not yet appended to messages)
    answer = state.get("proposed_answer") or get_last_ai_message_content(
        state["messages"]
    )
    documents = state["documents"]

    cited_answer = f"{answer}\n\n**Sources:**\n"
    for i, doc in enumerate(documents):
        source_url = doc.metadata.get("source", "N/A")
        cited_answer += f"[{i + 1}] {source_url}\n"

    # Append a single assistant message (no replacement)
    return {
        "messages": [AIMessage(content=cited_answer)],
        "proposed_answer": None,
    }


def handle_retrieval_failure(state: GraphState) -> dict:
    logger.info("---NODE: HANDLE RETRIEVAL FAILURE---")
    return {
        "messages": [
            AIMessage(
                content="I'm sorry, but I couldn't find any information to answer your question, even after trying multiple strategies."
            )
        ]
    }


def handle_grounding_failure(state: GraphState) -> dict:
    logger.info("---NODE: HANDLE GROUNDING FAILURE---")
    return {
        "messages": [
            AIMessage(
                content="I found some information, but I could not construct a factually grounded answer. Please try rephrasing your question."
            )
        ]
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
    Performs dual fresh retrievals, applies RRF fusion, de-duplication,
    cross-encoder re-ranking, and diversity filtering.
    """
    logger.info("---NODE: SMART RETRIEVAL & RE-RANK (GROUNDING CORRECTION)---")

    # Build two effective queries
    original_query = get_last_human_message_content(state["messages"])
    transformed_query = state.get("transformed_query")

    # If transformed_query is missing, perform inline transformation
    if not transformed_query:
        logger.info("Performing inline query transformation with drift-avoidance guard")
        transformed_query = inline_transform_query(original_query)
        logger.info(f"Inline transformed query: {transformed_query}")

    # Perform two fresh retrievals
    logger.info("Performing dual fresh retrievals")

    # First retrieval with original query
    retriever1, client1 = create_retriever()
    try:
        original_docs = retriever1.invoke(original_query)
        logger.info(f"Original query retrieved {len(original_docs)} documents")
    finally:
        client1.close()

    # Second retrieval with transformed query
    retriever2, client2 = create_retriever()
    try:
        transformed_docs = retriever2.invoke(transformed_query)
        logger.info(f"Transformed query retrieved {len(transformed_docs)} documents")
    finally:
        client2.close()

    # Document-level de-duplication before fusion
    original_docs = deduplicate_documents(original_docs)
    transformed_docs = deduplicate_documents(transformed_docs)

    # Apply Reciprocal Rank Fusion (RRF)
    fused_docs = apply_reciprocal_rank_fusion([original_docs, transformed_docs], k=60)

    # Document-level de-duplication after fusion
    fused_docs = deduplicate_documents(fused_docs)

    # Keep candidate pool for cross-encoder (e.g., top 8)
    candidate_pool = fused_docs[:8]
    logger.info(
        f"Selected top {len(candidate_pool)} candidates for cross-encoder re-ranking"
    )

    if not candidate_pool:
        logger.warning("No candidates available for cross-encoder re-ranking")
        return {"documents": [], "retrieval_success": False, "is_web_search": False}

    # Large cross-encoder re-ranking
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=settings.CROSS_ENCODER_MODEL_LARGE
    )
    # Use the transformed query for scoring (or original if transformation failed)
    scoring_query = transformed_query or original_query
    pairs = [[scoring_query, doc.page_content] for doc in candidate_pool]
    scores = cross_encoder.score(pairs)
    scored_docs = list(zip(candidate_pool, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in scored_docs]

    logger.info(f"Cross-encoder re-ranking complete for {len(reranked_docs)} documents")

    # Simple diversity filter after cross-encoder (final top_k selection)
    final_docs = apply_diversity_filter(
        reranked_docs, top_k=4, similarity_threshold=0.85
    )

    logger.info(
        f"Smart retrieval complete. Final selection: {len(final_docs)} documents"
    )

    return {
        "documents": final_docs,
        "retrieval_success": bool(final_docs),
        "is_web_search": False,
    }


def hybrid_context_retrieval(state: GraphState) -> dict:
    """
    Combines the best internal documents with fresh web search results.
    Includes document-level de-duplication when merging results.
    """
    logger.info("---NODE: HYBRID CONTEXT RETRIEVAL (GROUNDING CORRECTION)---")
    internal_documents = state["documents"]

    # Perform a web search
    web_search_state = web_search(state)
    web_documents = web_search_state.get("documents", [])

    # Combine documents
    combined_docs = internal_documents + web_documents
    logger.info(
        f"Combined {len(internal_documents)} internal docs with {len(web_documents)} web docs"
    )

    # Document-level de-duplication after combining
    deduplicated_docs = deduplicate_documents(combined_docs)

    logger.info(
        f"Hybrid context retrieval complete. Final count: {len(deduplicated_docs)} documents"
    )

    return {"documents": deduplicated_docs, "is_web_search": True}


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
