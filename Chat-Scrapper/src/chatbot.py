from dotenv import load_dotenv
load_dotenv()

from src.retriever import RAGRetriever

class CPChatbot:
    def __init__(self, retriever, system_message):
        self.retriever = retriever
        self.system_message = system_message

    def chat(self, query):
        results = self.retriever.retrieve(query)
        seen = set()
        filtered = []
        for (text_type, text), score in results:
            if text_type == "editorial":
                # Prefer to show editorials first
                continue
            if text not in seen and score < 1e10:
                filtered.append((text, score))
                seen.add(text)
        response = "Here are the most relevant results:\n"
        for i, (text, score) in enumerate(filtered):
            response += f"{i+1}. {text} (Score: {score:.2f})\n"
        return response

if __name__ == "__main__":
    import os
    from src.embeddings import CodeBERTEmbedder
    from src.vectorstore import VectorStore

    api_token = os.environ.get("HF_API_TOKEN")
    if not api_token:
        raise ValueError("Hugging Face API token not found in environment variable HF_API_TOKEN.")
    embedder = CodeBERTEmbedder(api_token)
    vector_store = VectorStore(dim=768)
    vector_store.load("data/vector_store.index", "data/vector_store_texts.json")
    retriever = RAGRetriever(embedder, vector_store)
    system_message = "You are a helpful assistant for Codeforces problems."
    chatbot = CPChatbot(retriever, system_message)

    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        print(chatbot.chat(query))

