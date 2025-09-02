from typing import List, Dict
import numpy as np
from loguru import logger
from langchain_core.documents import Document

from hyde.hyde import HyDE, CustomEncoder, CustomSearcher
from hyde.generator import OpenAIGenerator
from hyde.promptor import Promptor
from create_faissdb import load_vector_store, load_qa_dataset
import os
os.environ['http_proxy'] = 'http://localhost:7890'
os.environ['https_proxy'] = 'http://localhost:7890'

# KEY = 'sk-JJWo7wRovwXbgVl8gs2q1F4fns8sqOuPecxybkmCY0HEaOib'
KEY = 'sk-Il9wwtbnn9M4U9ZvJSImOYO8cPZeiXk7k37qdQMvD3VwgIh4'


class HyDERetriever:
    def __init__(self, vector_store, model_path):
        """
        初始化HyDE检索器

        Args:
            vector_store: FAISS向量存储
            model_path: 训练好的编码器模型路径
        """
        # 初始化各个组件
        self.promptor = Promptor(task='air_question')  # 可以根据需要选择不同的任务
        self.generator = OpenAIGenerator(
            model_name='gpt-3.5-turbo',
            api_key=KEY,
            base_url='base_url="https://xiaoai.plus/v1",'
        )
        self.encoder = CustomEncoder(model_path)
        self.searcher = CustomSearcher(vector_store)

        # 创建HyDE实例
        self.hyde = HyDE(
            promptor=self.promptor,
            generator=self.generator,
            encoder=self.encoder,
            searcher=self.searcher
        )

    def search(self, query: str, top_k: int = 1) -> list[Document]:
        """
        执行HyDE检索
        Args:
            query: 查询文本
            top_k: 返回的结果数量
        Returns:
            格式化的检索结果文本
        """

        # 执行端到端检索
        hits = self.hyde.con_search(query, k=top_k)

        return hits

        logger.error(f"HyDE search error: {e}")
        # 发生错误时回退到基本检索
        results = self.searcher.vector_store.similarity_search(query, k=top_k)
        return results

    # def embedding_search(self, query: str, top_k: int = 1) -> list:
    #     """
    #     执行HyDE检索
    #     Args:
    #         query: 查询文本
    #         top_k: 返回的结果数量
    #
    #     Returns:
    #         格式化的检索结果文本
    #     """
    #     try:
    #         # 执行端到端检索
    #         # 1. 查询预处理
    #         processed_query = self._preprocess_query(query)
    #         # 2. 生成多个假设文档并选择最佳的
    #         hypothesis_docs = self._generate_multiple_hypotheses(processed_query)
    #         # 3. 组合向量策略
    #         final_results = self._enhanced_search(processed_query, hypothesis_docs, top_k)
    #         # 4. 后处理和重排序
    #         reranked_results = self._rerank_results(query, final_results)
    #         return self._format_results(reranked_results)
    #         # hits = self.hyde.direct_search(query, k=top_k)
    #         # return hits
    #     except Exception as e:
    #         logger.error(f"HyDE search error: {e}")
    #         # 发生错误时回退到基本检索
    #
    #         # results = self.searcher.vector_store.similarity_search(query, k=top_k)
    #         # return results
    #
    # def _generate_multiple_hypotheses(self, query: str, num_hypotheses: int = 3) -> List[str]:
    #     """
    #     生成多个假设文档并选择最佳的
    #     """
    #     hypotheses = []
    #     prompts = [
    #         f"Generate a detailed technical description for: {query}",
    #         f"Explain the maintenance procedure for: {query}",
    #         f"Describe the troubleshooting steps for: {query}"
    #     ]
    #
    #     for prompt in prompts:
    #         try:
    #             hypothesis = self.generator.generate(prompt)
    #             hypotheses.append(hypothesis)
    #         except Exception as e:
    #             logger.warning(f"Failed to generate hypothesis for prompt: {prompt}")
    #             logger.error(e)
    #             continue
    #
    #     return hypotheses
    # def _enhanced_search(self, query: str, hypothesis_docs: List[str], top_k: int) -> List[Dict]:
    #     """
    #     增强的搜索策略
    #     """
    #     # 1. 编码所有文档
    #     query_vector = self.encoder.encode(query)
    #     hypothesis_vectors = [self.encoder.encode(doc) for doc in hypothesis_docs]
    #     # 2. 计算加权平均向量
    #     weights = [0.4, 0.3, 0.3]  # 可以根据实际效果调整权重
    #     combined_vector = np.average(
    #         [query_vector] + hypothesis_vectors,
    #         weights=[0.7] + [0.3 / len(hypothesis_vectors)] * len(hypothesis_vectors),
    #         axis=0
    #     )
    #     # 3. 执行搜索
    #     results = self.searcher.embedding_search(combined_vector, k=top_k * 2)  # 获取更多结果用于重排序
    #     return results
    # def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
    #     """
    #     重排序结果
    #     """
    #     # 1. 计算相似度分数
    #     scored_results = []
    #     for doc, score in results:
    #         # 计算文本相似度
    #         text_similarity = self._calculate_text_similarity(query, doc.page_content)
    #         # 组合分数
    #         final_score = 0.7 * score + 0.3 * text_similarity
    #         scored_results.append((doc, final_score))
    #
    #     # 2. 重排序
    #     return sorted(scored_results, key=lambda x: x[1], reverse=True)
    # @staticmethod
    # def _calculate_text_similarity(query: str, doc_text: str) -> float:
    #     """
    #     计算文本相似度
    #     """
    #     from sklearn.feature_extraction.text import TfidfVectorizer
    #     from sklearn.metrics.pairwise import cosine_similarity
    #
    #     vectorizer = TfidfVectorizer()
    #     try:
    #         tfidf_matrix = vectorizer.fit_transform([query, doc_text])
    #         return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    #     except Exception as e:
    #         logger.error(f"Error calculating text similarity: {e}")
    #
    #         return None
    # @staticmethod
    # def _format_results(results: List) -> List:
    #     """
    #     将重排序后的结果格式化为与 similarity_search 相同的格式
    #
    #     Args:
    #         results: List of (document, score) tuples from search
    #
    #     Returns:
    #         List of documents in the same format as similarity_search
    #     """
    #     formatted_results = []
    #     for doc, score in results:
    #         # 创建与 similarity_search 返回格式相同的文档对象
    #         formatted_doc = Document(
    #             page_content=doc.page_content,
    #             metadata=doc.metadata
    #         )
    #         formatted_results.append(formatted_doc)
    #
    #     return formatted_results
    # @staticmethod
    # def _preprocess_query(query: str) -> str:
    #     """
    #     查询预处理
    #     """
    #     # 1. 添加领域特定的关键词
    #     domain_keywords = ["aircraft", "maintenance", "procedure", "fault", "system"]
    #     enhanced_query = query
    #     for keyword in domain_keywords:
    #         if keyword.lower() in query.lower():
    #             enhanced_query = f"In aviation maintenance context: {query}"
    #             break
    #
    #     # 2. 扩展查询
    #     return enhanced_query


def main():
    # 配置参数
    faiss_index_path = 'database/faiss_new'
    model_path = "Models/st_model/best_model_0.690"
    test_data_path = 'dataset/retrieval_data/new/test.json'

    # 加载向量存储
    vector_store = load_vector_store(
        model_path=model_path,
        faiss_index_path=faiss_index_path,
        use_custom_model=True
    )

    # 创建HyDE检索器
    retriever = HyDERetriever(vector_store, model_path)

    # 测试单个查询
    query = "What is the initial evaluation of a patient with a suspected acute coronary syndrome?"
    results = retriever.search(query, top_k=1)
    print("Sample HyDE Search Results:")
    retrieved_questions = []
    for doc in results:
        print(type(doc))
        retrieved_questions.append(doc.metadata['question'])
    print(retrieved_questions)

    # 如果需要评估性能

    qa_dataset = load_qa_dataset(test_data_path)
    # metrics = evaluate_hyde_retrieval(vector_store, qa_dataset, top_k=1, model_path=model_path)
    #
    # print("\nHyDE Retrieval Metrics:")
    # for metric, value in metrics.items():
    #     print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()