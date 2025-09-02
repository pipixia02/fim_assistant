from typing import List, Dict
from sentence_transformers import CrossEncoder
from dataclasses import dataclass
from loguru import logger
import torch

from rerankers import Reranker
import cohere

@dataclass
class RankedDocument:
    """存储重排序后的文档及其分数"""
    page_content: str
    score: float
    metadata: Dict = None


class CrossEncoderReranker:
    def __init__(
            self,
            model_name: str = "./Models/reranker_model/best_model_0.902",
            device: str = None
    ):
        """初始化交叉编码器重排序器

        Args:
            model_name: 交叉编码器模型名称
            batch_size: 批处理大小
            device: 设备类型 ('cuda', 'cpu')
        """
        # 如果未指定device，自动检测
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.batch_size = 32
        logger.info(f"Loading cross-encoder model: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
            self,
            query: str,
            documents: List[Dict],
            top_k: int = None
    ) -> List[RankedDocument]:
        """重排序文档

        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_k: 返回前k个结果

        Returns:
            重排序后的文档列表
        """
        if not documents:
            logger.warning("Empty document list provided for reranking")
            return []

        # 准备文档对
        doc_pairs = [(query, doc.page_content) for doc in documents]
        try:
            # 批量预测相关性分数
            scores = self.model.predict(
                doc_pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            # 将文档和分数打包
            ranked_docs = [
                RankedDocument(
                    page_content=doc.page_content,
                    score=float(score),
                    metadata=doc.metadata,
                )
                for doc, score in zip(documents, scores)
            ]
            # 按分数降序排序
            ranked_docs.sort(key=lambda x: x.score, reverse=True)


            return ranked_docs

        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            return []

    def __call__(self, *args, **kwargs):
        """使对象可调用"""
        return self.rerank(*args, **kwargs)

    def save(self, path: str):
        """保存模型"""
        self.model.save(path)

    @classmethod
    def load(cls, path: str, device: str = None):
        """加载保存的模型"""
        model = cls(model_name=path, device=device)
        return model


def enhance_retrieval_with_reranking(query: str, vector_store, k: int = 10, top_k: int = 3):
    """增强检索系统

    Args:
        query: 用户查询
        vector_store: 向量存储
        k: 初始检索数量
        top_k: 重排序后保留数量
    """
    # 1. 初始化重排序器
    reranker = CrossEncoderReranker()

    # 2. 初始检索
    initial_results = vector_store.similarity_search(
        query,
        k=k  # 检索更多文档用于重排序
    )

    # 3. 重排序
    reranked_results = reranker.rerank(
        query=query,
        documents=initial_results,
        top_k=top_k
    )

    return reranked_results


if __name__ == '__main__':

    api_key = 'pcsk_7DddsD_HEtRRo8rnCFCM9E9LkFteKqhTzkdmTfkexSbLmZAmsc2aKDNSYU57XH9QChQz77'
    ranker = Reranker("pinecone", api_key=api_key, verbose=0)

    # ranker = Reranker("rankgpt3", api_key = 'sk-vbBVmZq6ZZWfX0y8SxUkNrE676Xxaw15WlvFnMGH9PiOp8uT')
    # ranker = Reranker("t5")
    # ranker = Reranker('./Models/reranker_model/best_model_0.894', model_type='cross-encoder')

    results = ranker.rank(query="I love you", docs=["I hate you", "I really like you"], doc_ids=[0,1])
    print(results)



    co = cohere.Client(api_key="<YOUR API KEY>")
    query = "What is the capital of the United States?"
    docs = ["Carson City is the capital city of the American state of Nevada. At the 2010 United States Census, Carson City had a population of 55,274.", "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.", "Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.", "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district. The President of the USA and many major national government offices are in the territory. This makes it the political center of the United States of America.", "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states. The federal government (including the United States military) also uses capital punishment."]
    results = co.rerank(model="rerank-english-v3.0", query=query, documents=docs, top_n=5, return_documents=True)
