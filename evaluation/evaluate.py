import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.retriever import create_retriever
from app.rag_chain import create_rag_chain
from app.logging_config import setup_logging, logger
from evaluation.ground_truth import get_ground_truth_data
from config import settings
import warnings

# Suppress the specific DeprecationWarning from ragas
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas.metrics.base")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas.metrics._context_recall")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas.metrics._context_precision")

# Initialize logging
setup_logging()

def run_evaluation():
    """
    Runs the RAGAS evaluation on the current RAG system using the latest API.
    """
    logger.info("--- Starting RAG Evaluation ---")

    # 1. Load Ground Truth Data
    ground_truth_data = get_ground_truth_data()
    questions = [item["question"] for item in ground_truth_data]
    ground_truths = [item["ground_truth_answer"] for item in ground_truth_data]

    # 2. Run the RAG Chain to Get Answers and Contexts
    answers = []
    contexts = []

    retriever, client = None, None
    try:
        retriever, client = create_retriever()
        rag_chain = create_rag_chain(retriever)

        for question in questions:
            logger.info(f"Generating answer for question: '{question}'")
            result = rag_chain.invoke(question)
            answers.append(result.content)

            # Retrieve the context used for this answer
            retrieved_docs = retriever.invoke(question)
            contexts.append([doc.page_content for doc in retrieved_docs])

    finally:
        if client and client.is_connected():
            client.close()
            logger.info("--- Weaviate connection closed ---")

    # 3. Prepare the Dataset for RAGAS
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    # 4. Wrap the LangChain models for RAGAS
    evaluation_llm = LangchainLLMWrapper(ChatOpenAI(model_name=settings.OPENAI_MODEL_NAME, openai_api_key=settings.OPENAI_API_KEY))
    # evaluation_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY))
    # Use the same embedding model as the one used in retriever for more consistent evaluation
    evaluation_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL))

    # 5. Run the Evaluation
    logger.info("--- Running RAGAS Evaluation ---")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=evaluation_llm,
        embeddings=evaluation_embeddings
    )

    # 6. Display the Results
    df = result.to_pandas()
    logger.info("--- RAGAS Evaluation Results ---")
    print(df.to_string())

    # Save the results to a CSV file for further analysis
    df.to_csv("ragas_evaluation_results.csv", index=False)
    logger.info("Evaluation results saved to ragas_evaluation_results.csv")


if __name__ == "__main__":
    run_evaluation()