from langchain_ollama import OllamaEmbeddings
import numpy as np

class OllamaEmbeddingsNorm(OllamaEmbeddings):
    def _process_emb_response(self, input: str) -> list[float]:
        emb = super()._process_emb_response(input)
        norm_1 = np.allclose(np.linalg.norm(emb), 1)
        if norm_1:
            return emb
        else:
            return (np.array(emb) / np.linalg.norm(emb)).tolist()


def get_embedding_handler(model, base_url):
    embeddings = OllamaEmbeddingsNorm(model=model, base_url=base_url)
    return embeddings