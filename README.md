# simple_rag

Perfect — you're building a **semantic document explorer** with **zero LLM**, powered by **embeddings + similarity + clustering + search + Streamlit UI**. Let’s break it down and build everything:

---

## 🔧 TECHNOLOGY STACK

Here’s what we’ll use:

| Tool                         | Purpose                                            |
| ---------------------------- | -------------------------------------------------- |
| **Python**                   | General scripting                                  |
| **ChromaDB**                 | Vector database for storing + searching embeddings |
| **SentenceTransformers**     | To embed paragraphs/sentences                      |
| **Scikit-learn (KMeans)**    | To cluster similar content                         |
| **Streamlit**                | For interactive web app                            |
| **Pandas**                   | Data handling                                      |
| *(Optional)* **UMAP/Plotly** | 2D cluster visualization                           |

---

## ✅ OVERVIEW OF THE PIPELINE

1. Load document (PDF or plain text)
2. Chunk into paragraphs/sentences
3. Embed each chunk
4. Cluster embeddings
5. Store in ChromaDB with metadata (`text`, `cluster_id`, etc.)
6. Build Streamlit interface:

   * Explore clusters by topic
   * Search semantically
   * Filter by clusters

---

## 📦 1. INSTALL REQUIRED LIBRARIES

```bash
pip install chromadb streamlit sentence-transformers scikit-learn pandas
```

*(Optional)* for PDF:

```bash
pip install PyMuPDF
```

---

## 🧠 2. PYTHON SCRIPT: CHUNK → EMBED → CLUSTER → STORE

```python
# prepare_chunks.py

import fitz  # PyMuPDF
import re
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# 1. Load and split text
def load_pdf_text(file_path):
    doc = fitz.open(file_path)
    return "\n".join(page.get_text() for page in doc)

def split_text_into_chunks(text, max_chunk_size=300):
    paragraphs = re.split(r"\n+", text)
    return [p.strip() for p in paragraphs if len(p.strip()) >= 20][:500]  # limit for demo

# 2. Embed chunks
def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(chunks)

# 3. Cluster embeddings
def cluster_embeddings(embeddings, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings)
    return cluster_ids

# 4. Store in ChromaDB
def store_in_chroma(chunks, embeddings, cluster_ids):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="document_chunks")

    ids = [str(i) for i in range(len(chunks))]
    metadatas = [
        {"text": chunks[i], "cluster": int(cluster_ids[i])}
        for i in range(len(chunks))
    ]

    collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)

    return collection

# MAIN
if __name__ == "__main__":
    text = load_pdf_text("sample.pdf")  # Replace with your file
    chunks = split_text_into_chunks(text)
    embeddings = embed_chunks(chunks)
    cluster_ids = cluster_embeddings(embeddings, num_clusters=7)
    store_in_chroma(chunks, embeddings, cluster_ids)
    print("✅ Done: chunks embedded, clustered, and stored.")
```

---

## 🖥️ 3. STREAMLIT FRONTEND APP

```python
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
```

---

## ✅ WHAT YOU NOW HAVE:

* 🔹 Fully **local pipeline**
* 🔹 Clustering without needing LLMs
* 🔹 Semantic search based on sentence-level relevance
* 🔹 Interactive **Streamlit app**

---

## 🧠 OPTIONAL ENHANCEMENTS:

| Feature                                 | How                                          |
| --------------------------------------- | -------------------------------------------- |
| **Topic labeling for clusters**         | Use keywords from chunks in cluster (TF-IDF) |
| **UMAP plot of clusters**               | `umap-learn` + `plotly`                      |
| **Search within selected cluster only** | Use `.query(..., where={"cluster": x})`      |
| **Auth / Upload new docs in Streamlit** | Add upload + process pipeline                |
| **Batch processing**                    | Run over many docs, tag by `doc_id`          |

---

## 🚀 FINAL THOUGHT

You’re now equipped with a **modular, local, LLM-free semantic retrieval system**, using embedding + clustering + vector search — great for research, enterprise docs, technical support, etc.

Would you like:

* A downloadable ZIP of the whole app?
* Or to add support for multiple documents / filtering by titles?

Let me know how you want to expand it.
