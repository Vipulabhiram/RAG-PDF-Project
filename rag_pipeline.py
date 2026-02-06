import fitz
import numpy as np
import faiss
import re
import os
import streamlit as st
from sentence_transformers import SentenceTransformer


def extract_text_from_pdf(pdf_source):
    if hasattr(pdf_source, "read"):
        pdf_bytes = pdf_source.read()
        pdf_source.seek(0)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    else:
        if not os.path.exists(pdf_source):
            raise FileNotFoundError(pdf_source)
        doc = fitz.open(pdf_source)

    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()
    return text



def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()



def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap

    return chunks



def build_faiss_index(chunks, model_name="all-MiniLM-L6-v2", progress_callback=None):
    if not chunks:
        raise ValueError("No chunks to index")

    if progress_callback:
        progress_callback("🔹 Loading embedding model...")

    model = SentenceTransformer(model_name)

    if progress_callback:
        progress_callback("🔹 Creating embeddings...")

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    if progress_callback:
        progress_callback("✅ FAISS index ready")

    return model, index, embeddings, chunks



def retrieve_context(query, model, index, chunks, top_k=5, score_threshold=0.25):
    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(query_emb, top_k)

    relevant_chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= score_threshold:
            relevant_chunks.append(chunks[idx])

    return relevant_chunks



def generate_answer(query, context, max_len=600):
    if "generator" not in st.session_state:
        raise RuntimeError("Text generator not initialized")

    if not context:
        return "❌ The answer is not found in the uploaded documents."

    generator = st.session_state.generator

    prompt = f"""
You are a document-based AI assistant.

RULES:
- Answer ONLY from the context
- If answer is missing, say: "The document does not contain this information."
- Do NOT use outside knowledge

Context:
{chr(10).join(context)}

Question:
{query}

Answer:
"""

    result = generator(
        prompt,
        max_length=max_len,
        temperature=0.0,
        do_sample=False
    )

    return result[0]["generated_text"].strip()
