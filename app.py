import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
from PyPDF2 import PdfReader
import os
import uuid

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_chunks")

# Sentence transformer model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Helper: Extract and chunk text from PDF
def extract_chunks(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
    return paragraphs

# Upload PDFs
st.title("🔍 Smart PDF Search Engine (No LLM)")
st.write("Upload your documents, search for similar content, and visualize clusters.")

uploaded_files = st.file_uploader("📁 Upload PDFs", type="pdf", accept_multiple_files=True)

# Import documents
if uploaded_files:
    for pdf_file in uploaded_files:
        paragraphs = extract_chunks(pdf_file)
        embeddings = model.encode(paragraphs).tolist()
        doc_title = pdf_file.name
        ids = [str(uuid.uuid4()) for _ in range(len(paragraphs))]
        metadatas = [{"title": doc_title, "chunk": i} for i in range(len(paragraphs))]
        collection.add(documents=paragraphs, ids=ids, embeddings=embeddings, metadatas=metadatas)
    st.success("✅ Documents embedded and added successfully!")

# Query interface
st.subheader("🔎 Query Text")
query_text = st.text_input("Enter text to search for similar content:")

if query_text:
    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)

    st.markdown("### 🧠 Top Matches:")
    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        st.write(f"**Title:** {meta['title']} | **Chunk:** {meta['chunk']} | **Similarity:** {1 - dist:.2f}")
        st.info(doc)

# View stored data
if st.checkbox("📚 View Stored Documents"):
    data = collection.get()
    titles = list(set([m['title'] for m in data['metadatas']]))
    selected_title = st.selectbox("Select Document", options=titles)
    filtered = [(doc, meta) for doc, meta in zip(data['documents'], data['metadatas']) if meta['title'] == selected_title]
    for doc, meta in filtered:
        st.markdown(f"**Chunk {meta['chunk']}**")
        st.write(doc)

# Cluster visualization
if st.checkbox("📊 Visualize Clusters"):
    st.markdown("### ✨ Document Embedding Clusters")
    data = collection.get()
    if len(data['embeddings']) < 3:
        st.warning("Not enough data to visualize.")
    else:
        dim_reduce = st.radio("Dimensionality Reduction", ["PCA", "t-SNE"])
        if dim_reduce == "PCA":
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, perplexity=5)

        reduced = reducer.fit_transform(data['embeddings'])
        df = {
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "title": [m['title'] for m in data['metadatas']],
            "chunk": [m['chunk'] for m in data['metadatas']],
            "text": data['documents']
        }

        fig = px.scatter(df, x="x", y="y", color="title", hover_data=["chunk", "text"])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🔍 Filter by Title to Search Within")
        selected_doc = st.selectbox("Choose Title", options=list(set(df['title'])))
        if selected_doc:
            sub_docs = [(t, d) for t, d in zip(df['chunk'], df['text']) if d in df['text'] and t in df['chunk']]
            for i, d in sub_docs:
                st.markdown(f"**Chunk {i}**")
                st.write(d)
