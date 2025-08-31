# ui/streamlit_app.py

import streamlit as st
import requests
import sys
import os

# Add the project's 'src' directory to the Python path
# This is necessary for the logging.ini to find the custom formatter
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from agentic_rag.logging_config import setup_logging

# --- Setup Logging ---
setup_logging()

st.set_page_config(page_title="Agentic RAG System", layout="wide")
st.title("📄 Agentic RAG System")

# Initialize session_id in Streamlit's session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Input box for the user query
query = st.text_input("Ask a question about your documents:", "")

if query:
    payload = {"query": query, "session_id": st.session_state.session_id}
    try:
        with st.spinner("Thinking..."):
            # Call the FastAPI backend
            response = requests.post("http://localhost:8000/query", json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes

            result = response.json()

            st.session_state.session_id = result["session_id"]

            st.success("Answer:")
            st.write(result["answer"])

            # Display token usage in an expander
            with st.expander("Show Token Usage"):
                st.write(f"Prompt Tokens: {result['prompt_tokens']}")
                st.write(f"Completion Tokens: {result['completion_tokens']}")
                st.write(f"Total Tokens: {result['total_tokens']}")

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the backend API: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")