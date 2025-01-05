import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import List, TypedDict
import unicodedata
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, StateGraph

def sanitize_text(text):
    normalized_text = unicodedata.normalize('NFKD', text)
    sanitized_text = ''.join(
        char for char in normalized_text
        if unicodedata.category(char)[0] not in ['C', 'M']
    )
    try:
        return sanitized_text.encode('ascii', errors='ignore').decode('ascii')
    except Exception:
        return sanitized_text.encode('utf-8', errors='ignore').decode('utf-8')

def load_webpage_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.content, 'html.parser')
        content_classes = ["post-content", "post-title", "post-header"]
        extracted_text = []
        for cls in content_classes:
            elements = soup.find_all(class_=cls)
            for elem in elements:
                clean_text = sanitize_text(elem.get_text(strip=True))
                if clean_text:
                    extracted_text.append(clean_text)
        final_text = "\n\n".join(extracted_text)
        return sanitize_text(final_text)
    except Exception as e:
        st.error(f"Error loading webpage: {e}")
        return ""

def initialize_rag_system(api_key, url="https://lilianweng.github.io/posts/2023-06-23-agent/"):
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
        vector_store = InMemoryVectorStore(embeddings)
        webpage_text = load_webpage_content(url)
        docs = [Document(page_content=webpage_text, metadata={"source": url})]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        _ = vector_store.add_documents(documents=all_splits)
       
        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str
       
        prompt = hub.pull("rlm/rag-prompt")
       
        def retrieve(state: State):
            retrieved_docs = vector_store.similarity_search(state["question"])
            return {"context": retrieved_docs}
       
        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = prompt.invoke({"question": state["question"], "context": docs_content})
            response = llm.invoke(messages)
            return {"answer": response.content}
       
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
       
        return graph, st.success("RAG system initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None, None

def main():
    st.set_page_config(page_title="OpenAI RAG Q&A", page_icon="🤖")
    st.title("🔍 LangGraph Webpage RAG Assistant")
    st.markdown("Ask questions about the loaded document using OpenAI's models!")
    st.sidebar.header("🔑 OpenAI Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    url = st.sidebar.text_input(
        "Document URL",
        value="https://lilianweng.github.io/posts/2023-06-23-agent/",
        help="Enter a URL to load documents from"
    )
   
    if st.sidebar.button("Initialize RAG System"):
        if api_key:
            graph, _ = initialize_rag_system(api_key, url)
            if graph:
                st.session_state.rag_graph = graph
        else:
            st.warning("Please enter your OpenAI API Key")
   
    question = st.text_input("Enter your question:")
   
    if question:
        if 'rag_graph' in st.session_state and st.session_state.rag_graph:
            try:
                with st.spinner("Generating response..."):
                    response = st.session_state.rag_graph.invoke({"question": question})
                st.subheader("Answer:")
                st.write(response["answer"])
                with st.expander("Retrieved Context"):
                    for doc in response.get("context", []):
                        st.markdown(f"**Document Snippet:**\n{doc.page_content}")
            except Exception as e:
                st.error(f"Error processing question: {e}")
        else:
            st.warning("Please initialize the RAG system first by entering your API key and clicking 'Initialize RAG System'")

if __name__ == "__main__":
    main()