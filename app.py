# app.py

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Semantic Document Explorer", layout="wide")

# Init Chroma & Embedder
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="document_chunks")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.title("📄 Semantic Document Explorer (No LLM)")
st.write("Explore clustered insights or search semantically.")

# 1. Browse clusters
st.subheader("🔍 Browse by Cluster")
all_data = collection.get(include=["metadatas"])

# Extract clusters
clusters = sorted(set([m["cluster"] for m in all_data["metadatas"]]))
selected_cluster = st.selectbox("Select a cluster", clusters)

if selected_cluster is not None:
    cluster_chunks = [m["text"] for m in all_data["metadatas"] if m["cluster"] == selected_cluster]
    st.write(f"### 📁 Cluster {selected_cluster} - {len(cluster_chunks)} chunks")
    for c in cluster_chunks:
        st.markdown(f"• {c}")

st.divider()

# 2. Semantic Search
st.subheader("🧠 Semantic Search")
query = st.text_input("Enter your question or phrase")

if query:
    query_vec = embedder.encode([query])
    results = collection.query(query_embeddings=query_vec, n_results=5)
    st.write("### Top Matches:")
    for m in results["metadatas"][0]:
        st.markdown(f"• {m['text']} (Cluster {m['cluster']})")
