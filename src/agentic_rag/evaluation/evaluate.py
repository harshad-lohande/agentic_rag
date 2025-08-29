# src/agentic_rag/evaluation/evaluate.py

import pandas as pd
from datasets import Dataset
from ragas import evaluate
# Import the specific metric classes
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import warnings

from agentic_rag.app.retriever import create_retriever
from agentic_rag.app.rag_chain import create_rag_chain
from agentic_rag.logging_config import setup_logging, logger
from agentic_rag.evaluation.ground_truth import get_ground_truth_data
from agentic_rag.config import settings
from agentic_rag.app.llm_provider import get_llm

# Suppress the internal RAGAS deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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

    # 4. Initialize metrics with the required models
    evaluation_llm = LangchainLLMWrapper(get_llm(fast_model=True))
    evaluation_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL))
    
    # Explicitly create and configure each metric
    faithfulness_metric = Faithfulness(llm=evaluation_llm)
    answer_relevancy_metric = AnswerRelevancy(llm=evaluation_llm, embeddings=evaluation_embeddings)
    context_recall_metric = ContextRecall(llm=evaluation_llm)
    context_precision_metric = ContextPrecision(llm=evaluation_llm)


    # 5. Run the Evaluation
    logger.info("--- Running RAGAS Evaluation ---")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness_metric,
            answer_relevancy_metric,
            context_recall_metric,
            context_precision_metric,
        ],
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