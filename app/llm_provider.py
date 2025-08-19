from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from config import settings

def get_llm(fast_model: bool = False):
    """
    Returns an LLM instance based on the provider specified in the settings.
    
    Args:
        fast_model: If True, returns a faster, potentially less powerful model.
    """
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "google":
        model_name = settings.GOOGLE_FAST_MODEL_NAME if fast_model else settings.GOOGLE_MAIN_MODEL_NAME
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.7,
        )
    elif provider == "openai":
        model_name = settings.OPENAI_FAST_MODEL_NAME if fast_model else settings.OPENAI_MAIN_MODEL_NAME
        return ChatOpenAI(
            model_name=model_name,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.7,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def get_embeddings():
    """
    Returns an embeddings instance based on the provider specified in the settings.
    Note: For this project, we are keeping the local HuggingFace model for retrieval.
    This function is for the evaluation framework's internal needs.
    """
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "google":
        return GoogleGenerativeAIEmbeddings(
            model=settings.GOOGLE_EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )
    elif provider == "openai":
        return OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")