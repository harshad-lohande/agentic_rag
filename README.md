# **Agentic RAG System: A Production-Grade Conversational AI**

This repository contains the source code for a sophisticated, local-first Agentic Retrieval-Augmented Generation (RAG) system. The application is designed from the ground up to be a production-ready, containerized solution that can ingest custom documents, maintain multi-turn conversations, and provide factually-grounded, cited answers.

This project moves beyond simple RAG chains by implementing a stateful, multi-agent architecture using LangGraph. It includes advanced features like conversational memory, automatic history summarization for token efficiency, and a final grounding-and-safety check to prevent hallucinations and ensure trustworthiness.

---

## **Core Features**

* **Conversational Memory**: The agent remembers the context of previous interactions in a session, allowing for natural, multi-turn follow-up questions.  
* **Efficient History Summarization**: To handle long conversations without exceeding token limits, the agent automatically creates rolling summaries of the chat history, ensuring both context retention and cost efficiency.  
* **Grounding & Safety Agent**: Every answer is passed through a final validation node that fact-checks the generated response against the source documents to prevent hallucinations.  
* **Dynamic Citations**: The safety agent adds specific, numbered citations to the final answer, referencing the source documents that support its claims, which adds a layer of credibility and verifiability.  
* **Token Usage Tracking**: The system tracks and displays the token consumption (prompt, completion, and total) for every turn of the conversation, providing clear visibility into API costs.  
* **Local-First & Containerized**: The entire application, including the vector database and the backend API, runs locally in a containerized environment using Docker, ensuring consistency and portability.

---

## **System Architecture**

The application is composed of several services that are orchestrated by Docker Compose:

1. **Weaviate Vector Store**: A containerized vector database that stores the embedded document chunks for fast retrieval.  
2. **FastAPI Backend**: A containerized backend service that hosts the agentic workflow. It exposes a simple API for the user interface to interact with.  
3. **LangGraph Agent**: The core logic of the application, defined as a stateful graph. The graph orchestrates the flow of a query through several nodes: document retrieval, history summarization, answer generation, and a final safety check.  
4. **Streamlit UI**: A simple, local web interface that allows users to interact with the backend API in a conversational manner.

---

## **Tech Stack & Key Libraries**

* **Orchestration**: LangGraph, LangChain  
* **Backend**: FastAPI, Uvicorn  
* **Frontend**: Streamlit  
* **Containerization**: Docker, Docker Compose  
* **Vector Database**: Weaviate  
* **Dependency Management**: Poetry  
* **Key Python Libraries**:  
  * pydantic-settings: For managing configuration and secrets.  
  * sentence-transformers: For creating high-quality text embeddings.  
  * langchain-openai: For interacting with OpenAI's language models.  
  * pypdf, python-docx: For parsing documents.

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

1. **Add Your Documents**: Place the .pdf, .docx, or .txt files you want to chat with into the data/ directory.  
2. **Build and Start the Backend**: This command builds the backend Docker image and starts the FastAPI and Weaviate containers.  
  
   ```
   docker compose up --build -d
   ```
   

3. **Ingest Your Data**: Run this script to process your documents and load them into the vector database.  
   
   ```
   poetry run python app/ingestion.py
   ```

4. **Launch the UI**: Start the Streamlit user interface.  
   
   ```
   poetry run streamlit run ui.py
   ```

   Navigate to **http://localhost:8501** in your browser to start your conversation.

---

## **Future Ideas and Roadmap**

This project has a solid foundation that can be extended with even more powerful agentic capabilities.

* **Implement Query Rewriting**: Add a node to the graph that analyzes the user's query in the context of the conversation and rewrites it to be more optimal for retrieval.  
* **Add a Re-ranking Agent**: Implement a node that takes the initially retrieved documents and uses a more powerful Cross-Encoder model to re-rank them for relevance before passing them to the generator.  
* **Tool Use**: Extend the agent with the ability to use tools, such as performing a web search if the answer cannot be found in the provided documents.  
* **Persistent Memory**: Replace the current InMemorySaver with a persistent checkpointer (like PostgresSaver or RedisSaver) so that conversations can be continued across application restarts.  
* **Deployment**: Package and deploy the application to a scalable, production-ready environment like Google Cloud Run or AWS App Runner.
