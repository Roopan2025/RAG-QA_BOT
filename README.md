# 📚 RAG-QA Bot

A **Retrieval-Augmented Generation (RAG) Question Answering Bot** that allows users to ask questions about PDF documents and receive concise answers based on the information available in those documents.

The application uses **LangChain, LangGraph, ChromaDB, Hugging Face Embeddings, Groq LLM, and Streamlit** to build an end-to-end document question-answering system.

---

## 🚀 Features

* 📄 Ask questions about PDF documents
* 🔍 Semantic search using vector embeddings
* 🧠 Retrieval-Augmented Generation (RAG)
* 🤖 Groq LLM for answer generation
* 🔗 LangChain integration
* 🕸️ LangGraph-based LLM workflow
* 💾 ChromaDB vector database
* 🤗 Hugging Face sentence-transformer embeddings
* 📑 Displays source document and reference pages
* 💬 Streamlit chat interface
* ⚡ Cached embedding model and vector database
* 🔒 Answers can be restricted to retrieved document context

---

## 🏗️ Project Architecture

```text
                    ┌──────────────────┐
                    │      User        │
                    │  Ask a Question  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Streamlit     │
                    │   Chat Interface │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Hugging Face   │
                    │    Embeddings    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     ChromaDB     │
                    │ Vector Retrieval │
                    └────────┬─────────┘
                             │
                      Relevant Chunks
                             │
                             ▼
                    ┌──────────────────┐
                    │    LangGraph     │
                    │     Workflow     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Groq LLM      │
                    │ Answer Generation│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Answer + Source  │
                    │  + Page Number   │
                    └──────────────────┘
```

---

## 🛠️ Tech Stack

| Technology            | Purpose                         |
| --------------------- | ------------------------------- |
| Python                | Programming language            |
| Streamlit             | User interface                  |
| LangChain             | LLM and RAG framework           |
| LangGraph             | LLM workflow orchestration      |
| ChromaDB              | Vector database                 |
| Hugging Face          | Text embedding model            |
| Sentence Transformers | Embedding generation            |
| Groq                  | Large Language Model            |
| PyMuPDF               | PDF document loading            |
| Pandas                | Data processing                 |
| python-dotenv         | Environment variable management |
| uv                    | Python dependency management    |

---

## 📂 Project Structure

```text
RAG-QA_BOT/
│
├── data/
│   ├── Embedded System Notes.pdf
│   └── embedded-systems.pdf
│
├── docs/
│   └── ...
│
├── config/
│   └── config.py
│
├── ingestion.py
├── app.py
├── requirements.txt
├── pyproject.toml
├── .env
├── .gitignore
└── README.md
```

> File names may differ depending on your project structure.

---

## 🔄 RAG Pipeline

The project follows these major steps:

### 1. Document Loading

PDF documents are loaded from the `data/` directory.

```text
PDF
 ↓
PyMuPDF Loader
 ↓
Documents
```

### 2. Document Splitting

Large documents are divided into smaller chunks.

```text
Document
 ↓
Text Splitter
 ↓
Multiple Chunks
```

This allows the retriever to find only the relevant portions of a document.

### 3. Embedding Generation

Each text chunk is converted into a numerical vector using the Hugging Face embedding model.

Example model:

```text
all-MiniLM-L6-v2
```

```text
Text Chunk
     ↓
Embedding Model
     ↓
Vector
```

### 4. Vector Storage

The generated embeddings are stored in **ChromaDB**.

```text
Text Chunks
     +
Embeddings
     ↓
ChromaDB
```

### 5. Similarity Search

When the user asks a question, the question is converted into an embedding and compared against the stored document vectors.

```text
User Question
      ↓
Question Embedding
      ↓
Similarity Search
      ↓
Relevant Document Chunks
```

### 6. Context Construction

The retrieved chunks are combined into a context.

```text
Relevant Chunk 1
Relevant Chunk 2
Relevant Chunk 3
        ↓
     Context
```

### 7. LLM Answer Generation

The context and question are passed to the Groq LLM.

```text
Context + Question
       ↓
    Groq LLM
       ↓
    Answer
```

### 8. Source Reference

The application displays:

* Source document
* Reference page numbers

Example:

```text
Source Document: Embedded System Notes.pdf
Reference Page No: 3, 4
```

---

## ⚙️ Installation

### Prerequisites

Make sure you have installed:

* Python 3.10+
* Git
* uv

Check Python:

```bash
python --version
```

Check uv:

```bash
uv --version
```

---

## 📥 Clone the Repository

```bash
git clone <https://github.com/Roopan2025/RAG-QA_BOT>
```

Move into the project directory:

```bash
cd RAG-QA_BOT
```

---

## 🐍 Create Virtual Environment

Using `uv`:

```bash
uv venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

---

## 📦 Install Dependencies

If you already have a `pyproject.toml`:

```bash
uv sync
```

Or install the required packages:

```bash
uv add streamlit
uv add langchain
uv add langgraph
uv add langchain-chroma
uv add langchain-huggingface
uv add langchain-groq
uv add langchain-pymupdf4llm
uv add sentence-transformers
uv add python-dotenv
uv add pandas
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit your `.env` file to GitHub.

Add this to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
chroma_db/
```

---

## 📄 Add Documents

Place your PDF documents inside:

```text
data/
```

Example:

```text
data/
├── Embedded System Notes.pdf
└── embedded-systems.pdf
```

---

## 🗄️ Create the Vector Database

Run your ingestion script:

```bash
python ingestion.py
```

The ingestion process:

```text
PDF Documents
     ↓
Load Documents
     ↓
Split Documents
     ↓
Generate Embeddings
     ↓
Store in ChromaDB
```

Example output:

```text
Loaded 385 pages
Created 540 chunks
Embedding model: all-MiniLM-L6-v2
Vector database created successfully
```

---

## ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 💬 Example Questions

If your PDF contains Embedded Systems content, you can ask:

```text
What is an embedded system?
```

```text
What are the characteristics of embedded systems?
```

```text
What are the components of an embedded system?
```

```text
What are the applications of embedded systems?
```

The application returns an answer along with the source information.

---

## 🧠 LangGraph Workflow

The current LangGraph workflow is simple:

```text
START
  ↓
Model
  ↓
END
```

The model receives:

```text
System Prompt
      +
Retrieved Context
      +
User Question
```

and generates the final answer.

---

## ⚡ Performance Optimization

The application uses Streamlit resource caching:

```python
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME
    )
```

This prevents the embedding model from being loaded repeatedly during every Streamlit rerun.

The vector database and ChatGroq model can also be cached.

---

## 🔐 RAG Grounding

The assistant can be configured to answer only from the retrieved document context.

Example system prompt:

```text
You are a question-answering assistant for the provided documents.

Answer the question ONLY using the retrieved context.

Do not use outside knowledge.

If the answer cannot be found in the context, say:
"I don't know based on the provided documents."

Keep the answer concise.
```

## 🚧 Future Improvements

* [ ] Conversation-aware question rewriting
* [ ] Better retrieval thresholding
* [ ] Hybrid search
* [ ] Metadata filtering
* [ ] Reranking retrieved documents
* [ ] Streaming LLM responses
* [ ] Chat history management
* [ ] Multiple PDF upload support
* [ ] PDF preview
* [ ] Improved source citations
* [ ] Authentication
* [ ] Cloud deployment
* [ ] Docker support
* [ ] Automated evaluation of RAG responses

---

## 🎯 Learning Outcomes

Through this project, I learned how to:

* Build a RAG application
* Process PDF documents
* Split documents into chunks
* Generate text embeddings
* Store vectors in ChromaDB
* Perform semantic similarity search
* Integrate Hugging Face embeddings
* Use Groq LLMs
* Build workflows using LangGraph
* Create interactive applications using Streamlit
* Manage environment variables
* Debug retrieval and LLM issues
* Improve RAG response grounding

---

## 👨‍💻 Author

**Roopan G**


## ⭐ Acknowledgements

This project uses open-source technologies and libraries from the Python, LangChain, LangGraph, Hugging Face, ChromaDB, Streamlit, and Groq ecosystems.

---

## 📜 License

This project is intended for educational and learning purposes.
