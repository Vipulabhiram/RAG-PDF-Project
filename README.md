# AI Knowledge Assistant

A PDF-based Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions.  
Answers are generated strictly from the document content to avoid hallucinations.

## Tech Stack
Python, Streamlit, FAISS, SentenceTransformers, PyMuPDF (fitz), Google FLAN-T5

## Features
- Upload and process multiple PDF files  
- Semantic search using vector embeddings  
- Document-grounded question answering  
- Simple chat-based interface  

## Run
pip install -r requirements.txt  
streamlit run app.py
