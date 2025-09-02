from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import wordpunct_tokenize, word_tokenize
from nltk.corpus import stopwords
from create_faissdb import load_vector_store, model_path, faiss_index_path
from typing import List, Dict, Union, Tuple
import numpy as np
from loguru import logger
from langchain_core.documents import Document
from hyde.hyde import HyDE, CustomEncoder, CustomSearcher
from hyde.generator import OpenAIGenerator
from hyde.promptor import Promptor
nltk.data.path.append('/home/hongchang/llm_projects/shenhang/nltk_data')
KEY = 'sk-P8CCBqMp5sAqZ5asHW7GnLUEmW4FPwbT9lOuaZqQ28wr7rGK'
import os
os.environ['http_proxy'] = 'http://localhost:7890'
os.environ['https_proxy'] = 'http://localhost:7890'


class AdvancedHyDERetriever:
    def __init__(self, vector_store, model_path: str):
        """
        初始化高级HyDE检索器
        Args:
            vector_store: 向量存储
            model_path: 训练好的编码器模型路径
            encoder_path: BGE编码器路径
        """
        # 初始化基础组件
        self.bm25_retriever = None
        self.vector_store = vector_store

        # 初始化稀疏检索器
        self.stop_words = set(stopwords.words('english'))
        self.documents = self._prepare_documents()
        self.bm25 = self._initialize_bm25()
        logger.info(f"Loaded {len(self.documents)} documents from vector store.")

        self.promptor = Promptor(task='air_question')  # 可以根据需要选择不同的任务
        self.generator = OpenAIGenerator(
            model_name='gpt-3.5-turbo',
            api_key=KEY,
        )
        self.encoder = CustomEncoder(model_path)
        self.searcher = CustomSearcher(vector_store)
        # 创建HyDE实例
        self.hyde = HyDE(
            promptor=self.promptor,
            generator=self.generator,
            encoder=self.encoder,
            searcher=self.vector_store
        )
        logger.info("Initialized HyDE components")

    def _prepare_documents(self) -> List[Dict]:
        """从向量存储中提取文档"""
        docs = []
        # 获取所有文档
        all_docs = self.vector_store.docstore._dict

        # 遍历文档存储中的所有文档
        for doc_id, doc in all_docs.items():
            docs.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        return docs

    def _initialize_bm25(self) -> BM25Okapi:
        """初始化BM25检索器"""
        tokenized_docs = [
            self._tokenize_text(doc['content'])
            for doc in self.documents
        ]
        return BM25Okapi(tokenized_docs)

    def _tokenize_text(self, text: str) -> List[str]:
        """文本分词处理"""
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token not in self.stop_words]

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """关键词检索"""
        tokenized_query = self._tokenize_text(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取前top_k个最高分的文档
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'content': self.documents[idx]['content'],
                'metadata': self.documents[idx]['metadata'],
                'score': scores[idx]
            })
        return results

    def _generate_similar_texts(self, docs: List[Document], query: str) -> List[str]:
        """生成相似格式的文本"""
        prompt_template = """
        As an aircraft maintenance expert, generate a description for the following question:
        - Keep the numbered format "(1)", "(2)"
        - Maintain exact technical terms and procedures
        - Focus only on the most relevant information for the qusetion

        Reference:
        {references}

        Question:
        {query}

        Response:
        """
        try:
            # 合并参考文档内容
            references = "\n\n***Example***\n".join([doc['content'] for doc in docs])
            # 生成提示
            prompt = prompt_template.format(
                references=references,
                query=query
            )
            # 生成新文本
            generated_text = self.generator.generate(prompt)

            return [generated_text]

        except Exception as e:
            logger.error(f"Error generating similar text: {e}")
            return []

    def search(self, query: str, top_k: int = 1, bm25_k=2, gen_k = 1) -> (List[Document], List[str]):
        """
        执行高级HyDE检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
        Returns:
            检索结果
        """

        # 1. BM25检索
        bm25_results = self._keyword_search(query, top_k=bm25_k)
        logger.info(f"BM25 retrieved {len(bm25_results)} documents")

        # 2. 生成相似文本
        generated_texts = self._generate_similar_texts(bm25_results, query)
        logger.info(f"Generated {len(generated_texts)} similar texts {generated_texts}")

        # 3. 使用生成的文本进行检索
        results = self.vector_store.similarity_search(generated_texts[0], top_k)

        return results

        # # 3. 使用生成的文本进行检索
        # results = []
        # for text in generated_texts:
        #     # 使用HyDE进行检索
        #     hyde_results = self.hyde.direct_search(text, k=top_k)
        #     results.extend(hyde_results)
        #
        # # 4. 重排序结果
        # if len(results) > top_k:
        #     reranked_results = self._rerank_results(query, [(doc, 1.0) for doc in results])
        #     return self._format_results(reranked_results)[:top_k]
        #
        # return results



    # def embedding_search(self, query: str, top_k: int = 5) -> List[Document]:
    #     """
    #     执行增强的嵌入检索
    #     """
    #     try:
    #         # 1. 查询预处理
    #         processed_query = self._preprocess_query(query)
    #         # 2. BM25检索获取初始文档
    #         bm25_docs = self.bm25_retriever.get_relevant_documents(processed_query)
    #         # 3. 生成假设文档
    #         hypothesis_docs = self._generate_similar_texts(bm25_docs[:3], processed_query)
    #         # 4. 执行增强检索
    #         results = self._enhanced_search(processed_query, hypothesis_docs, top_k)
    #         # 5. 重排序
    #         reranked_results = self._rerank_results(query, results)
    #
    #         return self._format_results(reranked_results)
    #
    #     except Exception as e:
    #         logger.error(f"Error in embedding search: {e}")
    #         return self.hyde.direct_search(query, k=top_k)
    # def _enhanced_search(self, query: str, hypothesis_docs: List[str], top_k: int) -> List[Tuple[Document, float]]:
    #     """增强的搜索策略"""
    #     # 从 hyde_retrieval.py 复用代码
    #     query_vector = self.encoder.encode(query)
    #     hypothesis_vectors = [self.encoder.encode(doc) for doc in hypothesis_docs]
    #
    #     combined_vector = np.average(
    #         [query_vector] + hypothesis_vectors,
    #         weights=[0.5] + [0.5 / len(hypothesis_vectors)] * len(hypothesis_vectors),
    #         axis=0
    #     )
    #
    #     results = self.vector_store.similarity_search_with_score_by_vector(
    #         combined_vector,
    #         k=top_k * 2
    #     )
    #     return results
    # def _rerank_results(self, query: str, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    #     """重排序结果"""
    #     # 从 hyde_retrieval.py 复用代码
    #     scored_results = []
    #     for doc, score in results:
    #         text_similarity = self._calculate_text_similarity(query, doc.page_content)
    #         final_score = 0.7 * score + 0.3 * text_similarity
    #         scored_results.append((doc, final_score))
    #
    #     return sorted(scored_results, key=lambda x: x[1], reverse=True)
    # @staticmethod
    # def _preprocess_query(query: str) -> str:
    #     """查询预处理"""
    #     # 从 hyde_retrieval.py 复用代码
    #     domain_keywords = ["aircraft", "maintenance", "procedure", "fault", "system"]
    #     enhanced_query = query
    #     for keyword in domain_keywords:
    #         if keyword.lower() in query.lower():
    #             enhanced_query = f"In aviation maintenance context: {query}"
    #             break
    #     return enhanced_query
    # @staticmethod
    # def _calculate_text_similarity(query: str, doc_text: str) -> float:
    #     """计算文本相似度"""
    #     # 从 hyde_retrieval.py 复用代码
    #     from sklearn.feature_extraction.text import TfidfVectorizer
    #     from sklearn.metrics.pairwise import cosine_similarity
    #
    #     vectorizer = TfidfVectorizer()
    #     try:
    #         tfidf_matrix = vectorizer.fit_transform([query, doc_text])
    #         return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    #     except:
    #         return 0.0
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

model_path = "../Models/st_model/best_model_0.691"
faiss_index_path = '../database/faiss_index_dom'

if __name__ == '__main__':
    # 初始化检索器
    vectordb = load_vector_store(
        model_path=model_path,
        faiss_index_path=faiss_index_path,
        use_custom_model=True
    )
      # replace with your API key, it can be OpenAI api key or Cohere api key

    retriever = AdvancedHyDERetriever(
        vector_store=vectordb,
        model_path=model_path
    )

    # 执行检索
    query = "What is the procedure for engine start?"
    results = retriever.search(query, top_k=1, bm25_k=3)
    print(results)
    # 打印结果
    for doc in results:
        print(f"Content: {doc.page_content}\n")
