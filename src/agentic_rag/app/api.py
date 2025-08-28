# app/api.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from agentic_rag.app.agentic_workflow import (
    GraphState,
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
    increment_retrieval_retry_counter,
    increment_grounding_retry_counter,
    route_retrieval_correction,
    route_grounding_correction,
    handle_retrieval_failure,
    handle_grounding_failure,
    generate_answer,
    summarize_history,
    grounding_and_safety_check,
    route_after_generation,
    route_after_safety_check,
    smart_retrieval_and_rerank,
    hybrid_context_retrieval,
)
from langgraph.checkpoint.memory import InMemorySaver
import uuid
from agentic_rag.logging_config import setup_logging, logger

# --- Setup Logging ---
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- Building and compiling autonomous LangGraph app at startup ---")
    
    workflow = StateGraph(GraphState)

    # --- Add all ACTION nodes to the graph ---
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate_hyde_document", generate_hyde_document)
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve_docs", retrieve_documents)
    workflow.add_node("rerank_documents", grade_and_rerank_documents)
    workflow.add_node("summarize_history", summarize_history)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("safety_check", grounding_and_safety_check)
    workflow.add_node("web_search_safety_check", web_search_safety_check)
    
    # Nodes for the retrieval correction loop
    workflow.add_node("enter_retrieval_correction", increment_retrieval_retry_counter)
    workflow.add_node("handle_retrieval_failure", handle_retrieval_failure)

    # Nodes for the grounding correction loop
    workflow.add_node("enter_grounding_correction", increment_grounding_retry_counter)
    workflow.add_node("smart_retrieval", smart_retrieval_and_rerank)
    workflow.add_node("hybrid_context", hybrid_context_retrieval)
    workflow.add_node("handle_grounding_failure", handle_grounding_failure)
    
    # --- Set up the graph's edges and conditional routing ---
    workflow.set_entry_point("classify_query")
    
    workflow.add_conditional_edges(
        "classify_query",
        route_for_retrieval,
        {"transform_query": "transform_query", "retrieve": "retrieve_docs"}
    )
    
    workflow.add_edge("transform_query", "retrieve_docs")
    
    workflow.add_conditional_edges(
        "retrieve_docs", 
        route_after_retrieval, 
        {"rerank_documents": "rerank_documents", "enter_retrieval_correction": "enter_retrieval_correction"}
    )

    workflow.add_conditional_edges(
        "rerank_documents", 
        route_after_reranking, 
        {"summarize": "summarize_history", "enter_retrieval_correction": "enter_retrieval_correction"}
    )

    # --- Retrieval Self-Correction Loop ---
    workflow.add_conditional_edges(
        "enter_retrieval_correction",
        route_retrieval_correction,
        {
            "transform_query": "transform_query",
            "generate_hyde_document": "generate_hyde_document",
            "web_search": "web_search",
            "handle_retrieval_failure": "handle_retrieval_failure"
        }
    )
    workflow.add_edge("generate_hyde_document", "retrieve_docs")

    # --- Generation Path ---
    workflow.add_edge("summarize_history", "generate_answer")
    workflow.add_edge("web_search", "summarize_history") 
    
    workflow.add_conditional_edges(
        "generate_answer",
        route_after_generation,
        {"safety_check": "safety_check", "web_search_safety_check": "web_search_safety_check"}
    )
    
    # --- Advanced Grounding Self-Correction Loop ---
    workflow.add_conditional_edges(
        "safety_check", 
        route_after_safety_check, 
        {"END": END, "enter_grounding_correction": "enter_grounding_correction"}
    )
    workflow.add_conditional_edges(
        "enter_grounding_correction",
        route_grounding_correction,
        {
            "smart_retrieval": "smart_retrieval",
            "hybrid_context": "hybrid_context",
            "handle_grounding_failure": "handle_grounding_failure"
        }
    )
    workflow.add_edge("smart_retrieval", "generate_answer")
    workflow.add_edge("hybrid_context", "generate_answer")
    
    # --- Endpoints ---
    workflow.add_edge("handle_retrieval_failure", END)
    workflow.add_edge("handle_grounding_failure", END)
    workflow.add_edge("web_search_safety_check", END)


    # --- Add a memory checkpointer ---
    memory = InMemorySaver()

    app.state.langgraph_app = workflow.compile(checkpointer=memory)
    
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
async def query_endpoint(request: QueryRequest):
    """Receives a query and returns a grounded answer using the LangGraph workflow."""
    langgraph_app = app.state.langgraph_app
    session_id = request.session_id or str(uuid.uuid4())
    
    logger.info(f"Received query for session_id: {session_id}")
    
    inputs = {"messages": [HumanMessage(content=request.query)]}
    config = {"configurable": {"thread_id": session_id}}

    final_state = langgraph_app.invoke(inputs, config=config)
    
    answer = final_state['messages'][-1].content
    
    return QueryResponse(
        answer=answer,
        prompt_tokens=final_state['prompt_tokens'],
        completion_tokens=final_state['completion_tokens'],
        total_tokens=final_state['total_tokens'],
        session_id=session_id,
    )

@app.get("/health")
def health():
    return {"status": "ok"}