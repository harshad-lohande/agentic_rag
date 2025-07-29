from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI

from config import settings

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
        llm = ChatOpenAI(
            model_name=settings.OPENAI_MODEL_NAME,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.7
        )

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