import numpy as np
from sentence_transformers import SentenceTransformer
from torch import Tensor


class CustomEncoder:
    """自定义编码器，使用训练好的模型"""

    def __init__(self, model_path: str):
        self.model = SentenceTransformer(model_path)

    def encode(self, text: str) -> Tensor:
        return self.model.encode(text)


class CustomSearcher:
    """自定义搜索器，使用FAISS向量存储"""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def embedding_search(self, query: np.ndarray, k: int = 5):
        # 将查询向量转换为正确的格式
        if len(query.shape) == 1:
            query_vector = query.reshape(1, -1)
            # 使用FAISS进行搜索
            docs_and_scores = self.vector_store.similarity_search_with_score_by_vector(
                query_vector[0], k=k
            )
        else:
            docs_and_scores = self.vector_store.similarity_search_with_score_by_vector(
                query, k=k
            )
        return docs_and_scores

    def search(self, query: str, k: int = 5):
        docs = self.vector_store.similarity_search(query, k=k)
        return docs


class HyDE:
    def __init__(self, promptor, generator, encoder, searcher):
        self.promptor = promptor
        self.generator = generator
        self.encoder = encoder
        self.searcher = searcher
    
    def prompt(self, query):
        return self.promptor.build_prompt(query)

    def generate(self, query):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        return hypothesis_documents

    def search(self, query, k=10):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        hits = self.searcher.search(hypothesis_documents, k=k)
        return hits

    def con_search(self, query, k=10):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        hypothesis_documents = query+ ' ' + hypothesis_documents
        hits = self.searcher.search(hypothesis_documents, k=k)
        return hits

if __name__ == '__main__':

    model = SentenceTransformer("../Models/BAAI/bge-base-en-v1.5")
