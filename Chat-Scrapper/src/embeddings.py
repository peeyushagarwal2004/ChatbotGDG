import requests

class CodeBERTEmbedder:
    def __init__(self, api_token, model_name="microsoft/codebert-base"):
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def generate_embedding(self, text):
        payload = {"inputs": text}
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        output = response.json()
        if isinstance(output, dict) and "error" in output:
            raise RuntimeError(f"Hugging Face API error: {output['error']}")
        vec = output[0]
        if isinstance(vec[0], list):
            embedding = [float(sum(col) / len(col)) for col in zip(*vec)]
        else:
            embedding = [float(x) for x in vec]
        return embedding

    def batch_generate_embeddings(self, texts):
        return [self.generate_embedding(text) for text in texts]

