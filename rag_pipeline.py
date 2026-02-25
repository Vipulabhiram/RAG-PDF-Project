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




def chunk_text(text, chunk_size=700, overlap=120):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunks.append(" ".join(words[start:end]))

        start = end - overlap

    return chunks




@st.cache_resource
def load_embedding_model(model_name="all-MiniLM-L6-v2"):

    return SentenceTransformer(model_name)




def build_faiss_index(chunks, model_name="all-MiniLM-L6-v2", progress_callback=None):

    if not chunks:
        raise ValueError("No chunks to index")

    if progress_callback:
        progress_callback("🔹 Loading embedding model...")

    model = load_embedding_model(model_name)

    if progress_callback:
        progress_callback("🔹 Creating embeddings...")

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    if progress_callback:
        progress_callback("✅ FAISS index ready")

    return model, index, embeddings, chunks




def retrieve_context(query, model, index, chunks, top_k=5):

    query_emb = model.encode(
        [query.lower()],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(query_emb, top_k)

    relevant_chunks = [chunks[idx] for idx in indices[0] if idx >= 0]

    return relevant_chunks



def generate_answer(query, context, max_len=150):

    if "generator" not in st.session_state:
        raise RuntimeError("Text generator not initialized")

    if not context:
        return "❌ The answer is not found in the uploaded documents."

    generator = st.session_state.generator

    prompt = f"""
You are an AI assistant answering questions using provided documents.

Instructions:
- Use the context to answer the question.
- The wording may differ but infer the meaning.
- If the answer is clearly not present say:
"The document does not contain this information."

Context:
{chr(10).join(context[:3])}

Question:
{query}

Answer clearly:
"""

    result = generator(
        prompt,
        max_new_tokens=max_len,
        temperature=0.0,
        do_sample=False
    )

    return result[0]["generated_text"].strip()