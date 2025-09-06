# src/agentic_rag/app/compression.py

from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    LLMChainExtractor,
)
from langchain_community.document_transformers.embeddings_redundant_filter import (
    EmbeddingsRedundantFilter,
)
from langchain.text_splitter import TokenTextSplitter

from agentic_rag.config import settings
from agentic_rag.logging_config import logger
from agentic_rag.app.hf_chat import HFChatInference

# OPTIONAL: centralize the numeric cap
COMPRESSION_MAX_NEW_TOKENS = int(getattr(settings, "COMPRESSION_MAX_TOKENS", 200) or 200)

def _resolve_hf_token() -> Optional[str]:
    return (
        getattr(settings, "HF_API_TOKEN", None)
        or getattr(settings, "HUGGINGFACEHUB_API_TOKEN", None)
        or None
    )


def get_compression_llm():
    prov = getattr(settings, "COMPRESSION_LLM_PROVIDER", "hf_endpoint")

    if prov in ("hf_endpoint", "hf_chat"):
        logger.info(
            f"--- Using HuggingFace LLM for compression: {getattr(settings, 'HF_COMPRESSION_MODEL', 'mistralai/Mistral-7B-Instruct-v0.3')} ---"
        )
        try:
            return HFChatInference(
                model=getattr(
                    settings, "HF_COMPRESSION_MODEL", "Qwen/Qwen2.5-14B-Instruct"
                ),
                token=_resolve_hf_token(),
                temperature=0.2,
                max_tokens=COMPRESSION_MAX_NEW_TOKENS,
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize HFChatInference for compression: {e}",
                exc_info=True,
            )

    elif prov == "ollama":
        logger.info(
            f"--- Using Ollama LLM for compression: {getattr(settings, 'COMPRESSION_LLM_MODEL', 'llama3.1:8b')} ---"
        )
        try:
            from langchain_ollama import ChatOllama
        except Exception:
            from langchain_community.chat_models import ChatOllama  # fallback
        return ChatOllama(
            base_url=getattr(settings, "OLLAMA_HOST", "http://localhost:11434"),
            model=getattr(settings, "COMPRESSION_LLM_MODEL", "qwen2.5:14b"),
            temperature=0.2,
            num_ctx=4096,
            num_predict=COMPRESSION_MAX_NEW_TOKENS, # Hard limit: max new tokens per compressed snippet
        )

    else:
        from agentic_rag.app.llm_provider import get_llm
        llm = get_llm(fast_model=True)
        try:
            llm.temperature = 0.2
        except Exception:
            pass

        if prov == "openai":
            logger.info(
                f"--- Using OpenAI LLM for compression: {getattr(settings, 'OPENAI_FAST_MODEL_NAME', 'gpt-4.1-nano')} ---"
            )
            llm.max_tokens = COMPRESSION_MAX_NEW_TOKENS
            return llm
        elif prov == "google":
            logger.info(
                f"--- Using Google Gemini LLM for compression: {getattr(settings, 'GOOGLE_FAST_MODEL_NAME', 'gemini-2.5-flash-lite')} ---"
            )
            llm.max_output_tokens = COMPRESSION_MAX_NEW_TOKENS
            return llm


def _token_splitter() -> Optional[TokenTextSplitter]:
    try:
        return TokenTextSplitter(
            encoding_name="cl100k_base",
            chunk_size=settings.COMPRESSION_MAX_TOKENS,
            chunk_overlap=settings.COMPRESSION_OVERLAP_TOKENS,
        )
    except Exception:
        approx = settings.COMPRESSION_MAX_TOKENS * 4
        over = settings.COMPRESSION_OVERLAP_TOKENS * 4
        return TokenTextSplitter(
            encoding_name=None, chunk_size=approx, chunk_overlap=over
        )


def build_document_compressor() -> DocumentCompressorPipeline:
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    redundant_filter = EmbeddingsRedundantFilter(
        embeddings=embeddings,
        similarity_threshold=settings.COMPRESSION_REDUNDANCY_SIM,
    )

    extractor_llm = get_compression_llm()
    extractor = LLMChainExtractor.from_llm(extractor_llm)

    steps = [redundant_filter, extractor]
    if settings.COMPRESSION_MAX_TOKENS > 0:
        steps.append(_token_splitter())

    return DocumentCompressorPipeline(transformers=steps)