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
