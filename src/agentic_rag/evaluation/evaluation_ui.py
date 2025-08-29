# src/agentic_rag/evaluation/evaluation_ui.py

import streamlit as st
import pandas as pd
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Evaluation Results",
    layout="wide",
)

st.title("📊 RAG Evaluation Results")

# --- Load and Display Evaluation Data ---
RESULTS_FILE = "ragas_evaluation_results.csv"

if os.path.exists(RESULTS_FILE):
    # Load the evaluation results from the CSV file
    df = pd.read_csv(RESULTS_FILE)

    st.header("Evaluation Metrics")
    st.write(
        """
        This table shows the RAGAS evaluation metrics for our RAG pipeline.
        - **Faithfulness**: How factually accurate is the answer?
        - **Answer Relevancy**: How relevant is the answer to the question?
        - **Context Recall**: Does the retriever fetch all necessary information?
        - **Context Precision**: Is the retrieved context signal greater than the noise?
        """
    )

    # Display the metrics in a clean table
    st.dataframe(df, use_container_width=True)

    # --- Display a summary of the results ---
    st.header("Results Summary")
    summary_df = df[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]].mean().to_frame('Average Score')
    st.dataframe(summary_df)


else:
    st.warning("No evaluation results file found.")
    st.info("Please run the evaluation first using the following command:")
    st.code("poetry run python -m evaluation.evaluate")