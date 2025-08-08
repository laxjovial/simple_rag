# app_logic.py
"""
This script contains all the core logic and functions for the Smart PDF Search Engine app.
It handles ChromaDB initialization, PDF processing, embedding, clustering, and querying.
"""

import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Initialize the Sentence Transformer model once to avoid re-loading
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def initialize_chromadb(collection_name: str = "document_chunks"):
    """Initializes the ChromaDB client and gets or creates a collection."""
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name=collection_name)
    return collection

def extract_chunks_from_pdf(pdf_file) -> tuple:
    """Extracts text from a PDF file and splits it into paragraphs (chunks)."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    # Split text by newlines and filter out short or empty paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
    return paragraphs

def add_documents_to_db(collection, pdf_file, paragraphs: list) -> None:
    """
    Embeds document chunks and adds them to the ChromaDB collection.
    
    Args:
        collection: The ChromaDB collection object.
        pdf_file: The uploaded PDF file object.
        paragraphs: A list of text chunks from the PDF.
    """
    if paragraphs:
        embeddings = MODEL.encode(paragraphs).tolist()
        doc_title = pdf_file.name
        ids = [str(uuid.uuid4()) for _ in range(len(paragraphs))]
        metadatas = [{"title": doc_title, "chunk": i} for i in range(len(paragraphs))]
        
        # Add the documents to the collection
        collection.add(
            documents=paragraphs, 
            ids=ids, 
            embeddings=embeddings, 
            metadatas=metadatas
        )

def query_db(collection, query_text: str, n_results: int = 5) -> dict:
    """
    Performs a similarity search in the ChromaDB collection.
    
    Args:
        collection: The ChromaDB collection object.
        query_text: The text to search for.
        n_results: The number of results to return.
    
    Returns:
        A dictionary of search results from ChromaDB.
    """
    if not query_text:
        return {}
    query_embedding = MODEL.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results

def get_all_data(collection):
    """Retrieves all documents, embeddings, and metadata from the collection."""
    return collection.get(include=["embeddings", "metadatas", "documents"])

def perform_dimensionality_reduction(embeddings: list, method: str = "PCA") -> list:
    """Reduces the dimensionality of embeddings for visualization."""
    if method == "PCA":
        reducer = PCA(n_components=2)
    elif method == "t-SNE":
        # Perplexity must be less than the number of samples
        num_samples = len(embeddings)
        perplexity = min(5, num_samples - 1) if num_samples > 1 else 1
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings.tolist()

def cluster_embeddings(embeddings: list, n_clusters: int = 5) -> list:
    """Performs K-Means clustering on the embeddings."""
    if len(embeddings) < n_clusters:
        return [0] * len(embeddings) # Assign all to a single cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)
    return cluster_ids.tolist()

