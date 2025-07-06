from src.embeddings import CodeBERTEmbedder
from src.vectorstore import VectorStore

class RAGRetriever:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query, k=5):
        query_embedding = self.embedder.generate_embedding(query)
        return self.vector_store.search(query_embedding, k)

