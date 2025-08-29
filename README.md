# **Autonomous Agentic RAG System: A Production-Grade Conversational AI**

This repository contains the source code for a sophisticated, local-first, and autonomous Retrieval-Augmented Generation (RAG) system. The application is designed from the ground up to be a production-ready, containerized solution that can ingest custom documents, maintain multi-turn conversations, and provide factually-grounded, cited answers.

This project moves beyond simple RAG chains by implementing a stateful, decision-driven, and self-correcting agentic architecture using LangGraph. The agent can dynamically alter its execution path based on the quality of its internal results, employing a series of advanced RAG techniques to recover from failures and improve the accuracy of its responses.

---

## **Core Features**

* **Autonomous & Multi-Layered Self-Correction**: The agent intelligently detects two distinct types of failures—**retrieval failure** and **grounding failure**—and triggers a dedicated, multi-step self-correction loop for each. This ensures the right recovery strategy is used for the right problem.
* **Adaptive RAG Strategies**: The agent has a toolkit of advanced RAG techniques it can deploy based on the specific failure:
    * **For Retrieval Failures**:
        * **Conversational Query Rewriting**: Transforms conversational follow-ups into precise, standalone queries.
        * **Hypothetical Document Embeddings (HyDE)**: Generates a hypothetical "perfect" answer to improve search relevance.
        * **Tool Use (Web Search)**: As a final fallback, uses the Tavily Search API to find answers if information is missing from local documents.
    * **For Grounding Failures**:
        * **Smart Retrieval & Re-ranking**: Deploys a more powerful cross-encoder model to perform a deeper, more accurate search of the internal knowledge base.
        * **Hybrid Context Synthesis**: Intelligently combines the best internal documents with fresh web search results to create a rich, synthesized context for generating a grounded answer.
* **Structured Output & Grounding**: The agent uses LLMs with structured output (Pydantic) to ensure reliable decision-making and to perform a final grounding check, programmatically adding citations and verifying the factual accuracy of every claim.
* **Local-First & Containerized**: The entire application, including the Weaviate vector database, runs locally in a containerized environment using Docker, ensuring data privacy, consistency, and portability.
* **Memory-Efficient Ingestion**: A streaming approach is used to parse and chunk large documents, allowing for the ingestion of very large files without overwhelming the system's memory.
* **Built-in Evaluation Framework**: The project includes an evaluation suite using the RAGAS framework to quantitatively measure the performance of the RAG pipeline.
* **Conversational Memory**: The agent remembers the context of previous interactions in a session, allowing for natural, multi-turn follow-up questions.
* **Structured Logging**: All application events are logged in a structured JSON format, making it easy to monitor, query, and analyze the system's behavior in a production environment.

---

## **System Architecture**

The application is built around a decision-driven agent orchestrated by LangGraph. Unlike a simple, linear workflow, this agent can dynamically route its execution based on the state of the conversation and the quality of its retrieved information.


The architecture includes:
1.  **Weaviate Vector Store**: A containerized vector database that stores the embedded document chunks for fast retrieval.
2.  **FastAPI Backend**: A containerized backend service that hosts the agentic workflow. It exposes a API for the user interface to interact with.
3.  **Autonomous LangGraph Agent**: The core of the application, defined as a stateful graph with conditional routing. The agent can classify queries, transform them, retrieve and re-rank documents, and enter a self-correction loop if necessary.
4.  **Streamlit UI**: A simple, web interface that allows users to interact with the backend API in a conversational manner.

---

## **Tech Stack & Key Libraries**

* **Orchestration**: LangGraph, LangChain
* **Backend**: FastAPI, Uvicorn
* **Frontend**: Streamlit
* **Containerization**: Docker, Docker Compose
* **Vector Database**: Weaviate
* **Dependency Management**: Poetry
* **Logging**: `python-json-logger`
* **Evaluation**: RAGAS
* **Key Python Libraries**:
    * `pydantic-settings`: For managing configuration.
    * `sentence-transformers`: For text embeddings and re-ranking.
    * `langchain-openai`, `langchain-google-genai`: For interacting with LLMs.
    * `tavily-python`: For the web search tool.

---

## **Continuous Integration (CI)**

This project includes a Continuous Integration (CI) pipeline using GitHub Actions to ensure code quality and stability. The pipeline automatically runs on every push and pull request to the main branch.

### **CI Pipeline Stages:**

1. **Lint & Format Check:** The code is automatically checked for style consistency and formatting errors using Ruff.

2. **Unit Testing:** Although I've carried out unit and integration tests locally and used placeholder for this step. You can push `tests/` directory containing tests you've written to the remote repo and make a slight change in `ci.yml` in `Run Unit Tests with Pytest` step to run  automated tests using Pytest to verify the correctness of individual as well as integrated components.

3. **(Roadmap) AI Quality Evaluation:** The pipeline is designed to be extended to automatically run the RAGAS evaluation suite, preventing deployments that would degrade the AI's answer quality.

This automated process ensures that all contributions are held to a high standard, making the project more robust and maintainable.

---

## **Setup and Installation (Ubuntu/Linux)**

This guide provides detailed instructions to set up the project from scratch.

### **1. Prerequisite Installation**

#### **Install Git**

```
sudo apt update
sudo apt install git
```

#### **Install Python 3.12**
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

#### **Install Poetry**

```
curl -sSL https://install.python-poetry.org | python3 -
# Add Poetry to your system's PATH as instructed by the installer
export PATH="/home/your-username/.local/bin:$PATH"
# Remember to add this line to your ~/.bashrc or ~/.zshrc file
```

#### **Install Docker and Docker Compose1**

```
# Set up Docker's repository
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to the docker group (requires logout/login to take effect)
sudo usermod -aG docker $USER
```

### **2. Project Setup**

#### **Step 1: Clone the Repository**

```
git clone <your-repository-url>
cd agentic-rag-system
```

#### **Step 2: Configure Python and Install Dependencies**

```
# Tell Poetry to use Python 3.12 for this project
poetry env use python3.12

# Install local dependencies for the UI and ingestion scripts
poetry install
```

#### **Step 3: Set Up Environment Variables**

Create a .env file in the root of the project and add your OpenAI API key.

```
touch .env
```

**.env file contents:**

Refer the .env.example

---

## **How to Run the Application 🚀**

1.  **Add Your Documents**: Place your `.pdf`, `.docx`, or `.txt` files into the `data/` directory.

2.  **Build and Start the Backend**: This command builds the backend Docker image and starts the FastAPI and Weaviate containers.
    ```bash
    docker compose up --build -d
    ```

3.  **Ingest Your Data**: Run this script to process your documents and load them into the vector database.
    ```bash
    poetry run python -m agentic_rag.scripts.run_ingestion
    ```
    or

    ```bash
    poetry run ingest
    ```


4.  **Launch the UI**: Start the Streamlit user interface.
    ```bash
    poetry run streamlit run ui/streamlit_app.py
    ```
    Navigate to **http://localhost:8501** in your browser to start your conversation.


5.  **Delete ingested data**: Run this script if you need to delete the indexed data.
    ```bash
    poetry run python -m agentic_rag.scripts.delete_indexed_data
    ```
    or

    ```bash
    poetry run delete-index
    ```

---

## **Evaluation**

This project includes a built-in evaluation framework using **RAGAS** (Retrieval-Augmented Generation Assessment) to quantitatively measure the performance of the RAG pipeline. This allows for a data-driven approach to improving the system and comparing the impact of different techniques.

The evaluation measures the following key metrics:

* **Faithfulness**: Measures how factually accurate the generated answer is based on the provided context.
* **Answer Relevancy**: Assesses how relevant the generated answer is to the user's question.
* **Context Precision**: Evaluates the signal-to-noise ratio of the retrieved context.
* **Context Recall**: Measures the ability of the retriever to fetch all the necessary information to answer the question.

### **How to Run the Evaluation**

1.  **Generate the Evaluation Results:**
    Run the following command from the root of the project. This will run the RAG chain against a ground truth dataset and save the results to a `ragas_evaluation_results.csv` file.

    ```bash
    poetry run python -m agentic_rag.evaluation.evaluate
    ```

2.  **Visualize the Results:**
    To view the results in a user-friendly web interface, run the following command:
    ```bash
    poetry run streamlit run src/agentic_rag/evaluation/evaluation_ui.py
    ```
    This will start a Streamlit application where you can view the evaluation metrics in a clean, interactive table.

---

## **Future Ideas and Roadmap**

* **Persistent Memory**: Replace the current `InMemorySaver` with a persistent checkpointer (like `PostgresSaver` or `RedisSaver`) so that conversations can be continued across application restarts.
* **Semantic Chunking**: Implement a more advanced chunking strategy based on semantic meaning rather than fixed sizes.
* **Deployment**: Package and deploy the application to a scalable, production-ready environment like Google Cloud Run or AWS App Runner.
