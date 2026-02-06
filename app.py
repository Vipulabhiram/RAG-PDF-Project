import streamlit as st
import time
import torch
from transformers import pipeline
from rag_pipeline import (
    extract_text_from_pdf,
    clean_text,
    chunk_text,
    build_faiss_index,
    retrieve_context,
    generate_answer
)


st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False

if "generator" not in st.session_state:
    device = 0 if torch.cuda.is_available() else -1
    st.session_state.generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device=device
    )


st.sidebar.header("📚 Upload PDF Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

process_button = st.sidebar.button("📌 Process Documents")
progress_box = st.sidebar.empty()


st.title("🤖 AI Knowledge Assistant")
st.write("Chat with your PDFs accurately — answers are grounded in your documents.")


if uploaded_files and process_button:
    progress_box.info("📖 Reading PDFs...")
    full_text = ""

    for pdf in uploaded_files:
        text = extract_text_from_pdf(pdf)
        full_text += clean_text(text) + " "

    progress_box.info("✂️ Chunking text...")
    chunks = chunk_text(full_text, chunk_size=300, overlap=50)

    def update_progress(msg):
        progress_box.info(msg)
        time.sleep(0.1)

    model, index, embeddings, chunk_list = build_faiss_index(
        chunks,
        progress_callback=update_progress
    )

    st.session_state.model = model
    st.session_state.index = index
    st.session_state.chunks = chunk_list
    st.session_state.docs_processed = True

    progress_box.success("✅ Documents processed successfully!")


def display_chat():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Bot:** {msg['content']}")

if st.session_state.docs_processed:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question about your documents")
        submitted = st.form_submit_button("Send")

        if submitted and user_input.strip():
            query = user_input.strip()

            st.session_state.messages.append(
                {"role": "user", "content": query}
            )

            context = retrieve_context(
                query=query,
                model=st.session_state.model,
                index=st.session_state.index,
                chunks=st.session_state.chunks,
                top_k=5,
                score_threshold=0.25
            )

            answer = generate_answer(
                query=query,
                context=context,
                max_len=600
            )

            st.session_state.messages.append(
                {"role": "bot", "content": answer}
            )

    display_chat()

else:
    st.info("⬅️ Upload PDFs and click **Process Documents** to start chatting.")


if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
