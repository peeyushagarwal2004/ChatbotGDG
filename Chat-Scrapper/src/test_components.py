from src.embeddings import CodeBERTEmbedder
from src.vectorstore import VectorStore
from src.retriever import RAGRetriever
from src.chatbot import CPChatbot

def test_embedder():
    embedder = CodeBERTEmbedder()
    test_text = "Find the maximum subarray sum"
    embedding = embedder.generate_embedding(test_text)
    print(f"Single embedding shape: {embedding.shape}")

    texts = [
        "Find the maximum subarray sum",
        "Implement a segment tree",
        "Solve the knapsack problem"
    ]
    embeddings = embedder.batch_generate_embeddings(texts)
    print(f"Batch embeddings shape: {embeddings.shape}")

def test_vector_store():
    vector_store = VectorStore(dim=768)
    embeddings = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
    texts = ["Text 1", "Text 2", "Text 3"]
    vector_store.add(embeddings, texts)
    results = vector_store.search([0.1] * 768, k=2)
    print("Search results:", results)

def test_retriever():
    embedder = CodeBERTEmbedder()
    vector_store = VectorStore(dim=768)
    vector_store.load("data/vector_store.index")
    retriever = RAGRetriever(embedder, vector_store)
    query = "How to solve the knapsack problem?"
    results = retriever.retrieve(query)
    print("Retriever results:", results)

def test_chatbot():
    embedder = CodeBERTEmbedder()
    vector_store = VectorStore(dim=768)
    vector_store.load("data/vector_store.index")
    retriever = RAGRetriever(embedder, vector_store)
    system_message = "I am solving a Competitive Programming problem. Help me understand its editorial."
    chatbot = CPChatbot(retriever, system_message)
    response = chatbot.chat("How do I solve problem C from Contest #792?")
    print("Chatbot response:", response)

if __name__ == "__main__":
    print("Testing Embedder...")
    test_embedder()
    print("\nTesting Vector Store...")
    test_vector_store()
    print("\nTesting Retriever...")
    test_retriever()
    print("\nTesting Chatbot...")
    test_chatbot()