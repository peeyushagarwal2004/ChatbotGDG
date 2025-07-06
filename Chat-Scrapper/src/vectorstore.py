import faiss
import numpy as np
import json

class VectorStore:
    def __init__(self, dim=768):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings).astype('float32'))
        self.texts.extend(texts)

    def save(self, index_path="data/vector_store.index", texts_path="data/vector_store_texts.json"):
        faiss.write_index(self.index, index_path)
        with open(texts_path, "w") as f:
            json.dump(self.texts, f)

    def load(self, index_path="data/vector_store.index", texts_path="data/vector_store_texts.json"):
        self.index = faiss.read_index(index_path)
        with open(texts_path, "r") as f:
            self.texts = json.load(f)

    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        results = []
        for j, i in enumerate(indices[0]):
            if 0 <= i < len(self.texts):
                results.append((self.texts[i], distances[0][j]))
        return results

