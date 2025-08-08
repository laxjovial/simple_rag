# app_ui.py
"""
This script defines the user interface for the Streamlit application.
It imports functions from app_logic.py to handle all the backend logic.
"""

import streamlit as st
import plotly.express as px
from app_logic import (
    initialize_chromadb,
    extract_chunks_from_pdf,
    add_documents_to_db,
    query_db,
    get_all_data,
    perform_dimensionality_reduction,
    cluster_embeddings
)

# Initialize ChromaDB connection
collection = initialize_chromadb()

def main():
    st.title("🔍 Smart PDF Search Engine (No LLM)")
    st.write("Upload your documents, search for similar content, and visualize clusters.")

    # --- Upload PDFs section ---
    st.subheader("📁 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing and embedding documents..."):
            for pdf_file in uploaded_files:
                paragraphs = extract_chunks_from_pdf(pdf_file)
                add_documents_to_db(collection, pdf_file, paragraphs)
        st.success("✅ Documents embedded and added successfully!")

    # --- Query Interface section ---
    st.subheader("🔎 Query Text")
    query_text = st.text_input("Enter text to search for similar content:")

    if query_text:
        results = query_db(collection, query_text)
        if results and results['documents']:
            st.markdown("### 🧠 Top Matches:")
            for doc, meta, dist in zip(
                results['documents'][0], results['metadatas'][0], results['distances'][0]
            ):
                st.write(
                    f"**Title:** {meta['title']} | **Chunk:** {meta['chunk']} | **Similarity:** {1 - dist:.2f}"
                )
                st.info(doc)
        else:
            st.warning("No matches found. Please upload documents first.")

    # --- View Stored Documents section ---
    st.subheader("📚 View Stored Documents")
    if st.checkbox("Show Stored Documents"):
        data = get_all_data(collection)
        if data['metadatas']:
            titles = sorted(list(set([m['title'] for m in data['metadatas']])))
            selected_title = st.selectbox("Select a Document to view:", options=titles)

            if selected_title:
                filtered_docs = [
                    (doc, meta) for doc, meta in zip(data['documents'], data['metadatas'])
                    if meta['title'] == selected_title
                ]
                for doc, meta in filtered_docs:
                    st.markdown(f"**Chunk {meta['chunk']}**")
                    st.write(doc)
        else:
            st.warning("No documents have been uploaded yet.")

    # --- Cluster Visualization section ---
    st.subheader("📊 Visualize Document Clusters")
    if st.checkbox("Show Cluster Visualization"):
        data = get_all_data(collection)
        embeddings = data['embeddings']
        metadatas = data['metadatas']
        documents = data['documents']

        if len(embeddings) < 2:
            st.warning("Not enough data to visualize. Please upload more documents.")
        else:
            dim_reduce_method = st.radio("Dimensionality Reduction Method", ["PCA", "t-SNE"])
            num_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)

            with st.spinner("Creating visualization..."):
                reduced_coords = perform_dimensionality_reduction(embeddings, dim_reduce_method)
                cluster_ids = cluster_embeddings(embeddings, num_clusters)

                df = {
                    "x": [c[0] for c in reduced_coords],
                    "y": [c[1] for c in reduced_coords],
                    "title": [m['title'] for m in metadatas],
                    "chunk": [m['chunk'] for m in metadatas],
                    "cluster": cluster_ids,
                    "text": documents
                }

                fig = px.scatter(
                    df,
                    x="x",
                    y="y",
                    color="title",
                    symbol="cluster",
                    hover_data=["chunk", "text"],
                    title="Document Embedding Clusters"
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
