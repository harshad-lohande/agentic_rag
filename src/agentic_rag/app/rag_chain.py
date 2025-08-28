# src/agentic_rag/app/rag_chain.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from agentic_rag.app.llm_provider import get_llm

def format_docs(docs):
    """Combines the content of multiple documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever: VectorStoreRetriever, llm=None):
    """
    Creates the RAG chain.
    Accepts an optional llm object for easier testing.
    """
    if llm is None:
        # If no LLM is provided, create the default one
        llm = get_llm(fast_model=True)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain