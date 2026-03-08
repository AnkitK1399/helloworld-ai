# Accounting RAG Retrieval System

This project implements a simple Retrieval-Augmented Generation (RAG) style system for searching accounting documents using semantic similarity.

Instead of keyword search, documents are converted into vector embeddings and stored in a FAISS vector database. User questions are also converted into embeddings and the system retrieves the most relevant documents.

## Technologies Used

- Python
- Sentence Transformers (Hugging Face)
- FAISS
- NumPy

## Project Structure

data/ → accounting text documents  
embeddings.py → converts documents to embeddings  
vector_store.py → stores vectors in FAISS  
rag_pipeline.py → retrieval pipeline  
app.py → command-line interface  

## Run the Project

Create virtual environment:
python -m venv venv
venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Run the system:
python app.py