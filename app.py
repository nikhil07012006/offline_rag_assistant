import streamlit as st
from models.model_loader import load_embedding_model, load_generation_model
from rag.vector_store import extract_text, create_chunks, build_faiss_index
from rag.generator import generate_answer
import numpy as np

st.title("Offline AI Document Intelligence System")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    embedder = load_embedding_model()
    tokenizer, model = load_generation_model()

    text = extract_text("temp.pdf")
    chunks = create_chunks(text)
    index, stored_chunks = build_faiss_index(chunks, embedder)

    question = st.text_input("Ask Question")

    if st.button("Get Answer"):

        query_embedding = embedder.encode([question])
        D, I = index.search(np.array(query_embedding), k=3)

        retrieved_text = " ".join([stored_chunks[i] for i in I[0]])

        answer = generate_answer(question, retrieved_text, tokenizer, model)

        st.write("### Answer")
        st.write(answer)

        st.write("### Retrieval Confidence")
        st.write(f"Similarity Score: {round(float(D[0][0]),2)}")
