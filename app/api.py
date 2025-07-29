from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from app.retriever import create_retriever
from app.rag_chain import create_rag_chain

# Use a lifespan event handler to load the model once at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Loading model and retriever at startup ---")
    retriever, client = create_retriever()
    app.state.rag_chain = create_rag_chain(retriever)
    app.state.weaviate_client = client
    yield
    # Clean up the models and connections
    print("--- Closing Weaviate connection at shutdown ---")
    if app.state.weaviate_client and app.state.weaviate_client.is_connected():
        app.state.weaviate_client.close()

app = FastAPI(lifespan=lifespan)

# Pydantic models for request and response
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Receives a query and returns a grounded answer with token stats."""
    rag_chain = app.state.rag_chain

    response = rag_chain.invoke(request.query)

    answer = response.content
    token_usage = response.response_metadata.get('token_usage', {})

    return QueryResponse(
        answer=answer,
        prompt_tokens=token_usage.get('prompt_tokens', 0),
        completion_tokens=token_usage.get('completion_tokens', 0),
        total_tokens=token_usage.get('total_tokens', 0)
    )

@app.get("/health")
def health():
    return {"status": "ok"}