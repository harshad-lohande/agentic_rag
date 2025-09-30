# ui/streamlit_app.py

import uuid

import requests
import streamlit as st

from agentic_rag.logging_config import setup_logging, logger
from agentic_rag.config import settings

# --- Setup Logging ---
setup_logging()

# --- Configuration ---
BACKEND_URL = settings.BACKEND_URL

# Securely access the API key from Streamlit's secrets manager
API_KEY = st.secrets.get("ENDPOINT_AUTH_API_KEY")

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Agentic RAG System", layout="wide")
st.title("Autonomous Agentic RAG System 🤖")

# Initialize session state for session_id and messages
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"New session created with ID: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Helper Functions ---
def display_chat_history():
    """Displays the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def query_backend(query: str, session_id: str) -> str:
    """
    Call backend and return answer string (never None).
    UI rendering happens in caller.
    """
    if not API_KEY:
        msg = "Backend API Key not configured. Set AUTH_API_KEY in .streamlit/secrets.toml"
        logger.error(msg)
        return msg

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,  # or Authorization: Bearer <API_KEY> if backend changed
    }
    payload = {"query": query, "session_id": session_id}

    try:
        with st.spinner("Thinking..."):
            resp = requests.post(BACKEND_URL, json=payload, headers=headers, timeout=120)
            status_code = resp.status_code
            if status_code != 200:
                # Try to extract backend error detail
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                logger.error(f"Backend HTTP {status_code}: {detail}")
                return f"Sorry, backend error ({status_code})."

            try:
                data = resp.json()
            except Exception as e:
                logger.error(f"JSON parse error: {e}")
                return "Sorry, invalid backend JSON."

            # Update session id if backend returns one
            if "session_id" in data:
                st.session_state.session_id = data["session_id"]

            # Prefer 'answer', fall back
            answer = data.get("answer") or data.get("response") or data.get("cached_answer")
            if not answer:
                logger.warning(f"No answer field in payload keys={list(data.keys())}")
                return "Sorry, backend returned no answer."

            # Store token stats for later display (optional)
            st.session_state.last_token_usage = {
                "prompt": data.get("prompt_tokens"),
                "completion": data.get("completion_tokens"),
                "total": data.get("total_tokens"),
            }
            return answer

    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {e}")
        return "Sorry, could not reach backend."


# --- Main Application Logic ---
display_chat_history()

if prompt := st.chat_input("Ask me anything about your documents..."):
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant answer
    answer_text = query_backend(prompt, st.session_state.session_id)

    # Add assistant answer to history
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    with st.chat_message("assistant"):
        st.markdown(answer_text)
        # Show token usage if available
        tu = st.session_state.get("last_token_usage")
        if tu and any(v is not None for v in tu.values()):
            with st.expander("Token Usage"):
                st.write(f"Prompt: {tu.get('prompt')}")
                st.write(f"Completion: {tu.get('completion')}")
                st.write(f"Total: {tu.get('total')}")