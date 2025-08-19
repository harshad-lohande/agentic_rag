from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from app.agentic_workflow import (
    GraphState,
    classify_query,
    transform_query,
    retrieve_documents,
    route_for_retrieval,
    route_for_generation,
    handle_retrieval_failure,
    generate_answer,
    summarize_history,
    grounding_and_safety_check
)
from langgraph.checkpoint.memory import InMemorySaver
import uuid
from app.logging_config import setup_logging, logger

# --- Setup Logging ---
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- Building and compiling autonomous LangGraph app at startup ---")
    
    workflow = StateGraph(GraphState)

    # --- Add the new nodes to the graph ---
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("retrieve_docs", retrieve_documents)
    workflow.add_node("handle_retrieval_failure", handle_retrieval_failure)
    workflow.add_node("summarize_history", summarize_history)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("safety_check", grounding_and_safety_check)

    # --- Set up the graph's edges and conditional routing ---
    workflow.set_entry_point("classify_query")
    
    # Conditional routing after classification
    workflow.add_conditional_edges(
        "classify_query",
        route_for_retrieval,
        {
            "transform_query": "transform_query",
            "retrieve": "retrieve_docs",
        }
    )
    
    workflow.add_edge("transform_query", "retrieve_docs")
    
    # Conditional routing after retrieval
    workflow.add_conditional_edges(
        "retrieve_docs",
        route_for_generation,
        {
            "summarize": "summarize_history",
            "retrieval_failure": "handle_retrieval_failure",
        }
    )

    workflow.add_edge("summarize_history", "generate_answer")
    workflow.add_edge("generate_answer", "safety_check")
    workflow.add_edge("safety_check", END)
    workflow.add_edge("handle_retrieval_failure", END)


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