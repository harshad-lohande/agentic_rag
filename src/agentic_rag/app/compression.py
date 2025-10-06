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
from agentic_rag.app.model_registry import model_registry

# OPTIONAL: centralize the numeric cap
COMPRESSION_MAX_NEW_TOKENS = settings.COMPRESSION_MAX_TOKENS


def _resolve_hf_token() -> Optional[str]:
    return settings.HUGGINGFACEHUB_API_TOKEN or None


def get_compression_llm():
    prov = settings.COMPRESSION_LLM_PROVIDER

    if prov in ("hf_endpoint", "hf_chat"):
        logger.info(
            f"--- Using HuggingFace LLM for compression: {settings.HF_COMPRESSION_MODEL} ---"
        )
        try:
            return HFChatInference(
                model=settings.HF_COMPRESSION_MODEL,
                token=_resolve_hf_token(),
                temperature=settings.COMPRESSION_MODEL_TEMP,
                max_tokens=COMPRESSION_MAX_NEW_TOKENS,
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize HFChatInference for compression: {e}",
                exc_info=True,
            )

    elif prov == "ollama":
        logger.info(
            f"--- Using Ollama LLM for compression: {settings.COMPRESSION_LLM_MODEL} ---"
        )
        try:
            from langchain_ollama import ChatOllama
        except Exception:
            from langchain_community.chat_models import ChatOllama  # fallback
        return ChatOllama(
            base_url=settings.OLLAMA_HOST,
            model=settings.COMPRESSION_LLM_MODEL,
            temperature=settings.COMPRESSION_MODEL_TEMP,
            num_ctx=settings.COMPRESSION_MODEL_CONTEXT_SIZE,
            num_predict=COMPRESSION_MAX_NEW_TOKENS,  # Hard limit: max new tokens per compressed snippet
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
                f"--- Using OpenAI LLM for compression: {settings.OPENAI_FAST_MODEL_NAME} ---"
            )
            llm.max_tokens = COMPRESSION_MAX_NEW_TOKENS
            return llm
        elif prov == "google":
            logger.info(
                f"--- Using Google Gemini LLM for compression: {settings.GOOGLE_FAST_MODEL_NAME} ---"
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
    # Use pre-loaded embedding model from registry for performance optimization
    embeddings = model_registry.get_embedding_model()
    if embeddings is None:
        # Fallback to on-demand loading if registry not initialized
        logger.warning(
            "Model registry not initialized for compression, loading embedding model on-demand"
        )
        embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    else:
        logger.debug(
            "Document compressor using pre-loaded embedding model from registry"
        )

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
