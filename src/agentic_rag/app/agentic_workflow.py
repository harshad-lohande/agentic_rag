# src/agentic_rag/app/agentic_workflow.py

import hashlib
import json
from typing import List, Literal, Dict, Tuple, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage

from agentic_rag.app.retriever import create_retriever
from agentic_rag.app.llm_provider import get_llm
from agentic_rag.app.semantic_cache import semantic_cache
from agentic_rag.app.model_registry import model_registry
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
from agentic_rag.app.compression import build_document_compressor
from agentic_rag.app.fast_compression import fast_compress_documents

# --- Pydantic Models for Structured Output ---


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


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
        description="The revised, fact-checked answer. If not grounded, this should be a message indicating failure.",
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
def extract_token_usage_from_response(response: Any) -> Dict[str, int]:
    """
    Safely extract token usage from a variety of LangChain response objects.
    Returns zeros when usage is unavailable (e.g., structured outputs that hide metadata).
    """
    # Typical LangChain generations
    meta = getattr(response, "response_metadata", None)
    if isinstance(meta, dict):
        usage = meta.get("token_usage") or meta.get("usage") or {}
        prompt = _as_int(usage.get("prompt_tokens", 0))
        completion = _as_int(usage.get("completion_tokens", 0))
        total = _as_int(usage.get("total_tokens", prompt + completion))
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }

    # Some wrappers keep original result under .raw
    raw = getattr(response, "raw", None)
    meta = getattr(raw, "response_metadata", None)
    if isinstance(meta, dict):
        usage = meta.get("token_usage") or meta.get("usage") or {}
        prompt = _as_int(usage.get("prompt_tokens", 0))
        completion = _as_int(usage.get("completion_tokens", 0))
        total = _as_int(usage.get("total_tokens", prompt + completion))
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }

    # Nothing usable; return zeros (no crash)
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def accumulate_token_usage(
    state: Dict[str, Any], new_usage: Dict[str, int]
) -> Dict[str, int]:
    """
    Returns a new dict with token counters added to the existing state totals.
    """
    return {
        "prompt_tokens": _as_int(state.get("prompt_tokens", 0))
        + _as_int(new_usage.get("prompt_tokens", 0)),
        "completion_tokens": _as_int(state.get("completion_tokens", 0))
        + _as_int(new_usage.get("completion_tokens", 0)),
        "total_tokens": _as_int(state.get("total_tokens", 0))
        + _as_int(new_usage.get("total_tokens", 0)),
    }


def _count_human_turns(messages: List[Any]) -> int:
    """Counts the number of Human messages in the conversation."""
    return sum(1 for m in messages if _msg_type(m) == "human")


def maybe_reset_usage_counters(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resets token counters at the start of a new super-step (new user turn).
    Detects a new user turn by counting Human messages.
    """
    messages = state.get("messages", [])
    current_turn_index = _count_human_turns(messages)
    last_index = state.get("token_usage_turn_index", None)

    if last_index is None or last_index != current_turn_index:
        logger.info(
            f"New super-step detected (human_turns {last_index} -> {current_turn_index}). Resetting token counters."
        )
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "token_usage_turn_index": current_turn_index,
        }

    # Keep current index if no reset necessary
    return {"token_usage_turn_index": current_turn_index}


# --- Parser helpers for raw-JSON prompts ---
def parse_query_classification(text: str) -> QueryClassification:
    """
    Parses classification from model output.
    Accepts JSON or plain text 'simple'/'complex'.
    """
    text = (text or "").strip()
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "classification" in data:
            val = str(data["classification"]).strip().lower()
            if val in ("simple", "complex"):
                return QueryClassification(classification=val)  # type: ignore
    except Exception:
        pass

    # Fallback to plain text
    tl = text.lower()
    if "complex" in tl and "simple" not in tl:
        return QueryClassification(classification="complex")
    if "simple" in tl and "complex" not in tl:
        return QueryClassification(classification="simple")
    # Heuristic default
    return QueryClassification(classification="simple")


def parse_grounding_check(text: str) -> GroundingCheck:
    """
    Parses grounding check from model output JSON.
    Falls back to heuristics if JSON is malformed.
    """
    text = (text or "").strip()
    try:
        data = json.loads(text)
        is_grounded = bool(data.get("is_grounded"))
        revised_answer = str(data.get("revised_answer", ""))
        return GroundingCheck(is_grounded=is_grounded, revised_answer=revised_answer)
    except Exception:
        tl = text.lower()
        is_grounded = ("true" in tl and "false" not in tl) or (
            "supported" in tl and "not" not in tl
        )
        return GroundingCheck(is_grounded=is_grounded, revised_answer=text)


# --- Helper functions for grounding correction improvements ---


def get_document_dedup_key(doc: Document) -> str:
    """
    Generate a de-duplication key based on CONTENT first (hash of normalized text).
    Fallback to (source, chunk_number) tuple if content is empty.
    """
    content = (doc.page_content or "").strip()
    if content:
        # Normalize whitespace so trivial formatting differences don't create distinct hashes
        normalized = " ".join(content.split())
        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return f"content:{content_hash}"

    # Fallback: (source, chunk_number) if both are present
    md = doc.metadata or {}
    src = md.get("source")
    cn = md.get("chunk_number")
    if src is not None and cn is not None:
        return f"{src}#{cn}"

    # Last-resort fallback: explicit id if present
    if md.get("id"):
        return f"id:{md['id']}"

    # Worst case: stable key from whatever source is available
    return hashlib.sha256((str(src) + str(cn)).encode("utf-8")).hexdigest()[:16]


def deduplicate_documents(documents: List[Document]) -> List[Document]:
    if not documents:
        return documents
    seen = set()
    deduplicated: List[Document] = []
    for d in documents:
        k = get_document_dedup_key(d)
        if k not in seen:
            seen.add(k)
            deduplicated.append(d)
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


def inline_transform_query(original_query: str) -> Tuple[str, Dict[str, int]]:
    """
    Inline transformation of query with drift-avoidance guard.
    Returns the rewritten query AND its token usage so callers can accumulate.
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
    usage = extract_token_usage_from_response(result)
    rewritten = getattr(result, "content", str(result))
    return rewritten, usage


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
    token_usage_turn_index: int | None = None  # NEW: for per-turn reset
    cache_hit: bool = False  # NEW: whether answer came from cache
    cache_query: str | None = None  # NEW: original query for caching
    cache_enabled: bool = settings.ENABLE_SEMANTIC_CACHE  # NEW: cache enablement flag


# --- Agentic Nodes ---


def _coerce_tavily_results_to_documents(raw: Any) -> List[Document]:
    """
    Normalize TavilySearch outputs into a list[Document], handling:
    - string
    - list[str]
    - list[dict] with keys like content/url or snippet/link
    - dict with 'results': [...]
    """
    if not raw:
        return []

    # If a single string summary
    if isinstance(raw, str):
        return [Document(page_content=raw, metadata={"source": "tavily"})]

    # If a dict, maybe has 'results'
    items: List[Any]
    if isinstance(raw, dict):
        if isinstance(raw.get("results"), list):
            items = raw["results"]
        else:
            items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return []

    docs: List[Document] = []
    for item in items:
        if isinstance(item, str):
            docs.append(Document(page_content=item, metadata={"source": "tavily"}))
            continue
        if isinstance(item, dict):
            content = (
                item.get("content")
                or item.get("snippet")
                or item.get("text")
                or item.get("title")
                or ""
            )
            url = item.get("url") or item.get("source") or item.get("link") or ""
            if not content:
                # last resort: serialize dict
                try:
                    content = json.dumps(item, ensure_ascii=False)
                except Exception:
                    content = str(item)
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": url or "tavily"},
                )
            )
    return docs


async def check_semantic_cache(state: GraphState) -> dict:
    """
    Check if there's a cached answer for the user's query.
    If found, return it directly and mark as cache hit.
    """
    query = get_last_human_message_content(state["messages"])
    
    if not query or not state.get("cache_enabled", True):
        logger.debug("Semantic cache disabled or no query found")
        return {"cache_hit": False, "cache_query": query}
    
    try:
        cached_result = await semantic_cache.get_cached_answer(query)
        
        if cached_result:
            # Cache hit - return cached answer
            cached_answer = cached_result["answer"]
            cached_metadata = cached_result.get("metadata", {})
            
            # Add cached answer as AI message
            ai_message = AIMessage(content=cached_answer)
            
            logger.info(f"Cache hit for query: {query[:50]}...")
            
            return {
                "messages": [ai_message],
                "cache_hit": True,
                "cache_query": query,
                "prompt_tokens": cached_metadata.get("prompt_tokens", 0),
                "completion_tokens": cached_metadata.get("completion_tokens", 0),
                "total_tokens": cached_metadata.get("total_tokens", 0),
                "retrieval_success": True,  # Mark as successful to avoid retrieval
                "grounding_success": True,  # Mark as grounded to avoid safety checks
            }
        else:
            # Cache miss - continue with normal flow
            logger.debug(f"Cache miss for query: {query[:50]}...")
            return {"cache_hit": False, "cache_query": query}
            
    except Exception as e:
        logger.error(f"Error checking semantic cache: {e}")
        return {"cache_hit": False, "cache_query": query}


async def store_in_semantic_cache(state: GraphState) -> dict:
    """
    Store the generated answer in semantic cache for future use.
    """
    if not state.get("cache_enabled", True) or state.get("cache_hit", False):
        # Don't cache if caching is disabled or this was already a cache hit
        return {}
    
    query = state.get("cache_query") or get_last_human_message_content(state["messages"])
    answer = get_last_ai_message_content(state["messages"])
    
    if not query or not answer:
        logger.debug("No query or answer to cache")
        return {}
    
    try:
        # Prepare metadata for caching
        cache_metadata = {
            "prompt_tokens": state.get("prompt_tokens", 0),
            "completion_tokens": state.get("completion_tokens", 0),
            "total_tokens": state.get("total_tokens", 0),
            "retrieval_retries": state.get("retrieval_retries", 0),
            "grounding_retries": state.get("grounding_retries", 0),
            "is_web_search": state.get("is_web_search", False),
            "documents_used": len(state.get("documents", [])),
        }
        
        success = await semantic_cache.store_answer(query, answer, cache_metadata)
        
        if success:
            logger.info(f"Cached answer for query: {query[:50]}...")
        else:
            logger.warning(f"Failed to cache answer for query: {query[:50]}...")
            
    except Exception as e:
        logger.error(f"Error storing answer in cache: {e}")
    
    return {}


def classify_query(state: GraphState) -> dict:
    """
    Classifies the user's query using raw generation to reliably capture token usage.
    Resets counters at the start of a new user turn (super-step).
    """
    logger.info("---NODE: CLASSIFY QUERY---")

    # Reset counters if this is a new user turn
    reset_patch = maybe_reset_usage_counters(state)

    last_message = get_last_message_content(state["messages"])
    conversation_history = format_messages_for_llm(state["messages"][:-1])

    prompt = ChatPromptTemplate.from_template(
        """You are an expert at analyzing conversations.
        Your task is to determine if the user's latest query is a simple, standalone question or 
        if it's a complex question that depends on the previous conversation history.

        Respond ONLY with JSON:
        {{
          "classification": "simple" | "complex"
        }}

        Conversation History:
        {history}

        User Query: {query}
        """
    )
    # Use raw LLM so we can capture token usage reliably
    llm = get_llm(fast_model=True)
    chain = prompt | llm

    result = chain.invoke({"history": conversation_history, "query": last_message})

    # Extract and accumulate token usage (accumulate against the reset state)
    new_tokens = extract_token_usage_from_response(result)
    logger.debug(f"classify_query token_usage extracted: {new_tokens}")
    totals = accumulate_token_usage({**state, **reset_patch}, new_tokens)

    # Parse structured result
    parsed = parse_query_classification(getattr(result, "content", str(result)))
    is_complex = parsed.classification == "complex"
    logger.info(f"Query classified as: {'complex' if is_complex else 'simple'}")

    # IMPORTANT: apply reset_patch FIRST, then totals so totals are not overwritten by zeros
    return {
        **reset_patch,
        **totals,
        "is_complex_query": is_complex,
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
    try:
        tool = TavilySearch(max_results=3)
        raw_results = tool.invoke(query)
        doc_objects = _coerce_tavily_results_to_documents(raw_results)
    except Exception as e:
        logger.error(f"Web search failed: {e}", exc_info=True)
        doc_objects = []

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
    try:
        documents = retriever.invoke(query)
    finally:
        client.close()
    dedup_documents = deduplicate_documents(documents)

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

    # Use pre-loaded cross-encoder model from registry for performance optimization
    cross_encoder = model_registry.get_cross_encoder_small()
    if cross_encoder is None:
        # Fallback to on-demand loading if registry not initialized
        logger.warning("Model registry not initialized for reranking, loading cross-encoder on-demand")
        cross_encoder = HuggingFaceCrossEncoder(
            model_name=settings.CROSS_ENCODER_MODEL_SMALL
        )
    else:
        logger.debug("Using pre-loaded small cross-encoder from registry")
        
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.score(pairs)
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    # Keep a few for compression to work with
    keep_n = max(1, getattr(settings, "RERANK_TOP_K", 3))
    reranked_docs = [doc for doc, score in scored_docs[:keep_n]]

    retrieval_success = bool(reranked_docs)
    logger.info(
        f"Re-ranking complete. Selected top {len(reranked_docs)} relevant documents."
    )

    return {"documents": reranked_docs, "retrieval_success": retrieval_success}


def compress_documents(state: GraphState) -> dict:
    """
    Compresses the (re)ranked documents with respect to the query.
    
    Uses fast extractive compression when ENABLE_FAST_COMPRESSION is True,
    otherwise falls back to the original LLM-based compression pipeline.
    """
    logger.info("---NODE: CONTEXTUAL COMPRESSION---")
    query = state.get("transformed_query") or get_last_message_content(
        state["messages"]
    )
    docs = state.get("documents", [])

    if not docs:
        logger.info("No documents to compress.")
        return {"documents": [], "retrieval_success": False}

    # Use fast compression if enabled for performance optimization
    if getattr(settings, 'ENABLE_FAST_COMPRESSION', True):
        logger.info("Using fast extractive compression for performance optimization")
        compressed_docs = fast_compress_documents(docs, query)
        compressed_docs = deduplicate_documents(compressed_docs)
    else:
        logger.info("Using original LLM-based compression pipeline")
        compressor = build_document_compressor()
        compressed_docs = compressor.compress_documents(docs, query=query)
        compressed_docs = deduplicate_documents(compressed_docs)
        logger.info(f"LLM compression: {len(docs)} docs → {len(compressed_docs)} snippets")

    return {"documents": compressed_docs, "retrieval_success": bool(compressed_docs)}


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
        turn_summary = getattr(summary_response, "content", str(summary_response))
        usage = extract_token_usage_from_response(summary_response)

        # Append the new summary to the existing one
        new_summary = f"{summary}\n- {turn_summary}".strip()

        totals = accumulate_token_usage(state, usage)
        return {
            "summary": new_summary,
            "turn_count": turn_count,
            **totals,
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
    usage = extract_token_usage_from_response(generation)
    totals = accumulate_token_usage(state, usage)

    # Only store the proposed answer here; do not touch messages
    return {
        "proposed_answer": getattr(generation, "content", str(generation)),
        **totals,
    }


def grounding_and_safety_check(state: GraphState) -> dict:
    """
    Final grounding check without adding citations.

    Behavior:
    - Validates whether the proposed answer is fully supported by the retrieved documents.
    - If grounded: appends a single assistant message with the (optionally lightly refined) answer.
    - If NOT grounded: does NOT append a message; downstream correction loop handles recovery.
    - No citation generation or source numbering is performed.
    """
    logger.info("---NODE: GROUNDING & SAFETY CHECK (NO-CITATIONS)---")
    question = get_last_human_message_content(state["messages"])
    answer = state.get("proposed_answer") or get_last_ai_message_content(
        state["messages"]
    )
    documents = state["documents"]

    grounding_prompt = ChatPromptTemplate.from_template(
        """You are a final grounding validator.

        Given:
        - A user question
        - A set of source documents (raw text)
        - A model-generated answer

        Your tasks:
        1. Determine whether EVERY factual claim in the answer is directly supported by the provided documents.
        2. If fully supported, return is_grounded=true and (optionally) a lightly edited answer for clarity. DO NOT add citations, reference markers, or fabricate information.
        3. If any claim is unsupported / hallucinated / contradicted, return is_grounded=false and a revised_answer that politely states you cannot answer confidently based on the provided documents (do NOT attempt to invent missing facts).

        Respond ONLY with a single JSON object:
        {{
        "is_grounded": true | false,
        "revised_answer": "string"
        }}

        Question:
        {question}

        Source Documents (unstructured):
        {context}

        Proposed Answer:
        {answer}
        """
    )

    llm = get_llm(fast_model=True)  # raw LLM for metadata
    grounding_chain = grounding_prompt | llm

    # Simple concatenation of documents (no numbering / citation formatting)
    def format_docs_plain(docs: List[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    response = grounding_chain.invoke(
        {
            "question": question,
            "context": format_docs_plain(documents),
            "answer": answer,
        }
    )

    # Token usage extraction & accumulation
    new_tokens = extract_token_usage_from_response(response)
    totals = accumulate_token_usage(state, new_tokens)

    parsed = parse_grounding_check(getattr(response, "content", str(response)))
    logger.info(f"Grounding check complete. is_grounded={parsed.is_grounded}")

    if parsed.is_grounded:
        # Append the grounded answer (no citations)
        return {
            "messages": [AIMessage(content=parsed.revised_answer)],
            "grounding_success": True,
            "proposed_answer": None,
            **totals,
        }
    else:
        # No message appended; triggers grounding correction path
        return {
            "grounding_success": False,
            **totals,
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
    Also accounts for token usage from inline_transform_query.
    """
    logger.info("---NODE: SMART RETRIEVAL & RE-RANK (GROUNDING CORRECTION)---")

    # Build two effective queries
    original_query = get_last_human_message_content(state["messages"])
    transformed_query = state.get("transformed_query")

    # If transformed_query is missing, perform inline transformation
    if not transformed_query:
        logger.info("Performing inline query transformation with drift-avoidance guard")
        transformed_query, usage = inline_transform_query(original_query)
        # Accumulate token usage from the inline transform
        totals = accumulate_token_usage(state, usage)
        state.update(
            totals
        )  # ensure subsequent additions start from the updated totals
        logger.info(f"Inline transformed query: {transformed_query}")
    else:
        totals = {
            "prompt_tokens": state.get("prompt_tokens", 0),
            "completion_tokens": state.get("completion_tokens", 0),
            "total_tokens": state.get("total_tokens", 0),
        }

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
        return {
            "documents": [],
            "retrieval_success": False,
            "is_web_search": False,
            **totals,
        }

    # Large cross-encoder re-ranking
    cross_encoder = model_registry.get_cross_encoder_large()
    if cross_encoder is None:
        # Fallback to on-demand loading if registry not initialized
        logger.warning("Model registry not initialized for smart retrieval, loading large cross-encoder on-demand")
        cross_encoder = HuggingFaceCrossEncoder(
            model_name=settings.CROSS_ENCODER_MODEL_LARGE
        )
    else:
        logger.debug("Using pre-loaded large cross-encoder from registry")
        
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
        **totals,
        "transformed_query": transformed_query,
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
        # proceed to contextual compression
        return "compress_documents"
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
        return "store_cache"  # Cache successful answers
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


def route_after_cache_check(state: GraphState) -> str:
    """
    Routes based on semantic cache check results.
    """
    logger.info("---ROUTER: ROUTE AFTER CACHE CHECK---")
    
    if state.get("cache_hit", False):
        logger.info("Cache hit - routing to cache storage and end")
        return "store_cache"
    else:
        logger.info("Cache miss - routing to query classification")
        return "classify_query"


def route_after_generation_with_cache(state: GraphState) -> str:
    """
    Enhanced routing after answer generation that includes cache storage.
    """
    logger.info("---ROUTER: ROUTE AFTER GENERATION WITH CACHE---")
    
    # First check if this needs safety/grounding check
    is_web_search = state.get("is_web_search", False)
    
    if is_web_search:
        logger.info("Web search answer - routing to web search safety check then cache")
        return "web_search_safety_check"
    else:
        logger.info("Internal answer - routing to safety check then cache")
        return "safety_check"
