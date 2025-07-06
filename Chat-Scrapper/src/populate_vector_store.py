import os
from dotenv import load_dotenv
from src.embeddings import CodeBERTEmbedder
from src.vectorstore import VectorStore
import numpy as np

load_dotenv()

def load_data(data_dir):
    texts = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r') as file:
            texts.append(("problem", file.read()))
    return texts

def main():
    api_token = os.environ.get("HF_API_TOKEN")
    if not api_token:
        raise ValueError("Hugging Face API token not found in environment variable HF_API_TOKEN.")
    embedder = CodeBERTEmbedder(api_token)
    vector_store = VectorStore(dim=768)

    problems = load_data("data/problems")
    editorials = load_data("data/editorials")
    metadata = load_data("data/metadata")
    texts = problems + editorials + metadata

    print(f"Loaded {len(texts)} texts.")
    if not texts:
        print("No texts found! Check your data folders.")
        return

    print("Generating embeddings via Hugging Face API...")
    embeddings = embedder.batch_generate_embeddings(texts)
    if not embeddings:
        print("No embeddings generated! Check your embedder and API token.")
        return

    embeddings = np.stack(embeddings).astype('float32')
    print("Embeddings shape:", embeddings.shape)

    print("Adding embeddings to vector store...")
    vector_store.add(embeddings, texts)
    vector_store.save("data/vector_store.index", "data/vector_store_texts.json")
    print("Vector store saved successfully!")

if __name__ == "__main__":
    main()