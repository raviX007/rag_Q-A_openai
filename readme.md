# **Retrieval-Augmented Generation (RAG) Q&A System**

This project implements a **Retrieval-Augmented Generation (RAG)** system to allow users to ask questions about content fetched from web pages. It uses **OpenAI Embeddings** and **FAISS Vector Database** to index and search document chunks, while leveraging **OpenAI GPT models** for generating accurate answers.

---

## **Table of Contents**

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Workflow Diagram](#workflow-diagram)
6. [How It Works](#how-it-works)
7. [License](#license)

---

## **Features**

- 📋 **Web Document Retrieval**: Fetches webpage content and sanitizes it for processing.
- 🔍 **FAISS Vector Search**: Uses FAISS for fast, efficient similarity search.
- 📊 **OpenAI Embeddings**: Converts document chunks into embeddings using OpenAI's `text-embedding-3-large` model.
- 🧠 **Q&A with GPT-4**: Generates answers with OpenAI GPT-4.
- ⚙️ **Interactive UI**: Built with **Streamlit** for an intuitive user experience.
- 🔑 **Configurable API Key**: Enter OpenAI API keys securely via the app interface.

---

## **Requirements**

Ensure you have the following dependencies installed:

- **Python 3.8+**
- **Streamlit**
- **Requests**
- **BeautifulSoup4**
- **LangChain**
- **LangGraph**
- **FAISS**
- **OpenAI SDK**

---

## **Installation**

1. **Clone the repository**:

   ```bash
   git clone https://github.com/raviX007/rag_Q-A_openai.git
   cd rag_Q-A_openai
   ```

2. **Create a virtual environment and activate it**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

---

## **Usage**

1. Start the Streamlit app.
2. Enter your **OpenAI API key** in the sidebar.
3. Provide a **URL** to fetch document content.
4. Initialize the **RAG system** by clicking the **"Initialize RAG System"** button.
5. Enter your question about the document in the input box.
6. View the generated answers and retrieved context snippets.

---

## **Project Structure**

The project follows a clean and modular structure:

![alt text](image.png)

---

## **How It Works**

### 1. **Document Retrieval**

- Fetch webpage content using `requests` and parse it with **BeautifulSoup**.
- Remove unnecessary HTML artifacts and sanitize the text.

### 2. **Text Processing**

- Split the document into smaller chunks using **LangChain**'s `RecursiveCharacterTextSplitter`.

### 3. **Vector Embeddings**

- Use **OpenAI Embeddings** (`text-embedding-3-large`) to convert chunks into vector representations.

### 4. **Vector Store (FAISS)**

- Store the vectorized chunks in **FAISS** for efficient similarity search.

### 5. **Retrieval-Augmented Generation**

- Perform similarity search on the FAISS vector store to retrieve the most relevant document chunks.
- Pass the retrieved context and user question to **OpenAI GPT-4** for answer generation.

### 6. **Interactive Q&A**

- Users can input questions, view answers, and see retrieved document snippets via the **Streamlit** UI.

---
