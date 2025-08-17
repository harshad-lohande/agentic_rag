# app/api.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from app.agentic_workflow import (
    GraphState,
    retrieve_documents,
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
    logger.info("--- Building and compiling LangGraph app at startup ---")
    
    workflow = StateGraph(GraphState)

    # Add the nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("summarize", summarize_history)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("safety_check", grounding_and_safety_check)

    # --- Update the graph's edges ---
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "summarize")
    workflow.add_edge("summarize", "generate")
    workflow.add_edge("generate", "safety_check")
    workflow.add_edge("safety_check", END)

    # --- Add a memory checkpointer ---
    memory = InMemorySaver()

    app.state.langgraph_app = workflow.compile(checkpointer=memory)
    
    logger.info("--- LangGraph app compiled successfully ---")
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