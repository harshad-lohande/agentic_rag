# src/agentic_rag/app/api.py

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Annotated
import uuid
import secrets

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from agentic_rag.app.agentic_workflow import (
    GraphState,
    check_semantic_cache,
    store_in_semantic_cache,
    classify_query,
    transform_query,
    generate_hyde_document,
    web_search,
    retrieve_documents,
    grade_and_rerank_documents,
    web_search_safety_check,
    route_for_retrieval,
    route_after_retrieval,
    route_after_reranking,
    route_after_cache_check,
    route_after_generation_with_cache,
    increment_retrieval_retry_counter,
    increment_grounding_retry_counter,
    route_retrieval_correction,
    route_grounding_correction,
    handle_retrieval_failure,
    handle_grounding_failure,
    generate_answer,
    summarize_history,
    grounding_and_safety_check,
    route_after_safety_check,
    smart_retrieval_and_rerank,
    hybrid_context_retrieval,
    compress_documents,
)
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from agentic_rag.logging_config import setup_logging, logger
from agentic_rag.config import settings
from agentic_rag.app.middlewares import RequestIDMiddleware
from agentic_rag.app.semantic_cache import semantic_cache
from agentic_rag.app.model_registry import model_registry
from agentic_rag.testing.semantic_cache_tester import (
    semantic_cache_tester,
    CacheEntryRequest,
    SimilarityTestRequest,
    SimilarityTestResponse,
    CacheTestResponse,
)

# --- Setup Logging ---
setup_logging()


# --- Authentication ---
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key_header: Annotated[str | None, Security(API_KEY_HEADER)]) -> str:
    """Validate X-API-Key header."""
    expected = settings.APP_ENDPOINT_API_KEY
    if not expected:
        logger.warning(
            "APP_ENDPOINT_API_KEY not set; rejecting request (set it in .env)"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key not configured",
        )
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )
    if not secrets.compare_digest(api_key_header, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return "ok"  # do not return the actual key


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- Building and compiling autonomous LangGraph app at startup ---")

    # --- Pre-load all ML models for performance optimization ---
    logger.info("Initializing model registry for performance optimization...")
    await model_registry.initialize_models()
    logger.info(
        "Model registry initialized - eliminating per-request model loading overhead"
    )

    # --- Use Redis for persistent, shareable state ---
    async with AsyncRedisSaver.from_conn_string(
        f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
    ) as checkpointer:
        # await checkpointer.asetup()

        workflow = StateGraph(GraphState)

        # --- Add all ACTION nodes to the graph ---
        workflow.add_node("check_cache", check_semantic_cache)
        workflow.add_node("store_cache", store_in_semantic_cache)
        workflow.add_node("classify_query", classify_query)
        workflow.add_node("transform_query", transform_query)
        workflow.add_node("generate_hyde_document", generate_hyde_document)
        workflow.add_node("web_search", web_search)
        workflow.add_node("retrieve_docs", retrieve_documents)
        workflow.add_node("rerank_documents", grade_and_rerank_documents)
        workflow.add_node("compress_documents", compress_documents)
        workflow.add_node("summarize_history", summarize_history)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("safety_check", grounding_and_safety_check)
        workflow.add_node("web_search_safety_check", web_search_safety_check)

        # Nodes for the retrieval correction loop
        workflow.add_node(
            "enter_retrieval_correction", increment_retrieval_retry_counter
        )
        workflow.add_node("handle_retrieval_failure", handle_retrieval_failure)

        # Nodes for the grounding correction loop
        workflow.add_node(
            "enter_grounding_correction", increment_grounding_retry_counter
        )
        workflow.add_node("smart_retrieval", smart_retrieval_and_rerank)
        workflow.add_node("hybrid_context", hybrid_context_retrieval)
        workflow.add_node("handle_grounding_failure", handle_grounding_failure)

        # --- Set up the graph's edges and conditional routing ---
        workflow.set_entry_point("check_cache")

        # Cache routing
        workflow.add_conditional_edges(
            "check_cache",
            route_after_cache_check,
            {"store_cache": "store_cache", "classify_query": "classify_query"},
        )

        workflow.add_conditional_edges(
            "classify_query",
            route_for_retrieval,
            {"transform_query": "transform_query", "retrieve": "retrieve_docs"},
        )

        workflow.add_edge("transform_query", "retrieve_docs")

        workflow.add_conditional_edges(
            "retrieve_docs",
            route_after_retrieval,
            {
                "rerank_documents": "rerank_documents",
                "enter_retrieval_correction": "enter_retrieval_correction",
            },
        )

        # --- When reranking succeeds, go to compression ---
        workflow.add_conditional_edges(
            "rerank_documents",
            route_after_reranking,
            {
                "compress_documents": "compress_documents",
                "enter_retrieval_correction": "enter_retrieval_correction",
            },
        )

        # --- After compression in the main path, proceed to summarization then generation ---
        workflow.add_edge("compress_documents", "summarize_history")
        workflow.add_edge("summarize_history", "generate_answer")

        # --- Retrieval Self-Correction Loop ---
        workflow.add_conditional_edges(
            "enter_retrieval_correction",
            route_retrieval_correction,
            {
                "transform_query": "transform_query",
                "generate_hyde_document": "generate_hyde_document",
                "web_search": "web_search",
                "handle_retrieval_failure": "handle_retrieval_failure",
            },
        )
        workflow.add_edge("generate_hyde_document", "retrieve_docs")

        # --- Generation Path ---
        workflow.add_edge("summarize_history", "generate_answer")
        workflow.add_edge("web_search", "summarize_history")

        workflow.add_conditional_edges(
            "generate_answer",
            route_after_generation_with_cache,
            {
                "safety_check": "safety_check",
                "web_search_safety_check": "web_search_safety_check",
            },
        )

        # --- Advanced Grounding Self-Correction Loop ---
        workflow.add_conditional_edges(
            "safety_check",
            route_after_safety_check,
            {
                "store_cache": "store_cache",
                "enter_grounding_correction": "enter_grounding_correction",
            },
        )
        workflow.add_conditional_edges(
            "enter_grounding_correction",
            route_grounding_correction,
            {
                "smart_retrieval": "smart_retrieval",
                "hybrid_context": "hybrid_context",
                "handle_grounding_failure": "handle_grounding_failure",
            },
        )
        # --- In grounding correction, pass through compression before generation ---
        workflow.add_edge("smart_retrieval", "compress_documents")
        workflow.add_edge("hybrid_context", "compress_documents")

        # --- Endpoints ---
        workflow.add_edge("store_cache", END)
        workflow.add_edge("handle_retrieval_failure", END)
        workflow.add_edge("handle_grounding_failure", END)
        workflow.add_edge("web_search_safety_check", "store_cache")

        app.state.langgraph_app = workflow.compile(checkpointer=checkpointer)
        logger.info("--- LangGraph app compiled successfully ---")

        # --- Save the graph as a Mermaid markdown file ---
        try:
            graph_mermaid = app.state.langgraph_app.get_graph(xray=True).draw_mermaid()
            with open("autonomous_rag_graph.md", "w") as f:
                f.write(graph_mermaid)
            logger.info("--- Graph visualization saved to autonomous_rag_graph.md ---")
        except Exception as e:
            logger.error(f"Failed to generate graph visualization: {e}")

        yield


logger.info("--- Application shutdown ---")


app = FastAPI(lifespan=lifespan)

# Add the RequestIDMiddleware to the application
app.add_middleware(RequestIDMiddleware)


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    session_id: str


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest, api_key: Annotated[str, Depends(get_api_key)]
):
    """Receives a query and returns a grounded answer using the LangGraph workflow."""
    langgraph_app = app.state.langgraph_app
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(f"Received query for session_id: {session_id}")

    inputs = {"messages": [HumanMessage(content=request.query)]}
    config = {"configurable": {"thread_id": session_id}, "checkpoint_ns": "dev"}
    # logger.info("--- View the information stored by checkpointer ---")
    # checkpoint_info = await langgraph_app.checkpointer.aget_tuple(config)
    # logger.info(checkpoint_info)

    final_state = await langgraph_app.ainvoke(inputs, config=config)

    answer = final_state["messages"][-1].content

    return QueryResponse(
        answer=answer,
        prompt_tokens=final_state["prompt_tokens"],
        completion_tokens=final_state["completion_tokens"],
        total_tokens=final_state["total_tokens"],
        session_id=session_id,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/cache/stats")
async def get_cache_stats():
    """Get semantic cache statistics."""
    return await semantic_cache.get_cache_stats()


@app.post("/cache/clear")
async def clear_cache():
    """Clear all cache entries."""
    success = await semantic_cache.clear_cache()
    return {
        "success": success,
        "message": "Cache cleared" if success else "Failed to clear cache",
    }


@app.get("/config/hnsw")
def get_hnsw_config():
    """Get current HNSW configuration."""
    return {
        "hnsw_ef_construction": settings.HNSW_EF_CONSTRUCTION,
        "hnsw_ef": settings.HNSW_EF,
        "hnsw_max_connections": settings.HNSW_MAX_CONNECTIONS,
        "index_name": settings.INDEX_NAME,
        "embedding_model": settings.EMBEDDING_MODEL,
    }


@app.get("/config/models")
def get_model_config():
    """Get current model registry configuration and status."""
    return model_registry.get_model_info()


# --- SEMANTIC CACHE TESTING ENDPOINTS (Remove before deploying to production) ---


@app.post("/cache/test/create-entry")
async def create_test_cache_entry(request: CacheEntryRequest):
    """
    Create a cache entry for testing purposes.

    This endpoint allows you to directly cache a query-answer pair
    without executing the entire graph workflow.
    """
    return await semantic_cache_tester.create_cache_entry(
        query=request.query, answer=request.answer, metadata=request.metadata
    )


@app.post("/cache/test/similarity", response_model=SimilarityTestResponse)
async def test_query_similarity(request: SimilarityTestRequest):
    """
    Test similarity between two queries using all available similarity methods.

    This endpoint compares a cached query with a test query using:
    - Vector similarity (if cached query exists in vector store)
    - Embedding similarity
    - Cross-encoder similarity
    - Lexical similarity

    It also predicts whether the test query would get a cache hit.
    """
    return await semantic_cache_tester.test_query_similarity(
        cached_query=request.cached_query, test_query=request.test_query
    )


@app.post("/cache/test/retrieval", response_model=CacheTestResponse)
async def test_cache_retrieval(query: str = None, request: Request = None):
    """
    Test cache retrieval for a given query.

    This endpoint tests whether a query would get a cache hit
    using the same logic as the actual workflow, without executing
    the full graph.

    Query can be provided either as a query parameter or in the request body.
    """
    # Try to get query from query parameters first
    if not query and request:
        query = request.query_params.get("query")

    # If still no query, try to get from request body
    if not query and request:
        try:
            body = await request.json()
            if isinstance(body, dict):
                query = body.get("query")
            elif isinstance(body, str):
                query = body
        except Exception as e:
            logger.error(f"Failed to parse request body: {e}")
            pass

    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    return await semantic_cache_tester.test_cache_retrieval(query)


@app.post("/cache/test/bulk-similarity")
async def bulk_similarity_test(cached_query: str, test_queries: list[str]):
    """
    Test similarity between one cached query and multiple test queries.

    This endpoint is useful for testing how a cached query performs
    against multiple variations or related queries.
    """
    return await semantic_cache_tester.bulk_similarity_test(cached_query, test_queries)


@app.get("/cache/test/stats")
async def get_cache_test_stats():
    """
    Get comprehensive cache statistics including testing framework information.

    This endpoint provides detailed cache statistics and information about
    the testing framework capabilities.
    """
    return await semantic_cache_tester.get_cache_statistics()


@app.post("/cache/test/clear")
async def clear_cache_for_testing():
    """
    Clear the cache for testing purposes.

    This endpoint clears all cache entries to start fresh testing.
    Use with caution as this affects the actual cache used by the system.
    """
    return await semantic_cache_tester.clear_cache_for_testing()
