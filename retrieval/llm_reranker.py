"""
使用Qwen2.5-7B-Instruct作为重排序器，对检索的候选文档进行排序
"""

import time
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from loguru import logger
from openai import OpenAI


@dataclass
class RankedDocument:
    """存储重排序后的文档及其分数"""
    page_content: str
    score: float
    metadata: Dict = None


class LLMReranker:
    """基于大语言模型的重排序器，使用Qwen2.5-7B-Instruct模型"""
    
    def __init__(
        self,
        model_name: str = "Qwen2.5-7B-Instruct-local",
        api_base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.1,
        max_tokens: int = 400,
        max_retries: int = 3,
        retry_delay: int = 5,
        scoring_mode: str = "relative"  # 'relative' 或 'absolute'
    ):
        """初始化LLM重排序器
        
        Args:
            model_name: 模型名称，默认为本地Qwen2.5-7B-Instruct
            api_base_url: API基础URL
            temperature: 温度参数
            max_tokens: 最大生成token数
            max_retries: 最大重试次数
            retry_delay: 重试延迟秒数
            scoring_mode: 评分模式，'relative'让LLM选择最佳文档，'absolute'让LLM给每个文档打分
        """
        self.model_name = model_name
        self.api_base_url = api_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.scoring_mode = scoring_mode
        
        # 初始化OpenAI客户端
        try:
            self.client = OpenAI(
                base_url=api_base_url,
                api_key="sk-xxx",  # 随便填写，只是为了通过接口参数校验
            )
            logger.info(f"LLM Reranker initialized with model: {model_name}, scoring mode: {scoring_mode}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Reranker: {e}")
            raise
    
    def call_llm_api(self, prompt: str) -> Optional[str]:
        """调用LLM API
        
        Args:
            prompt: 提示词
            
        Returns:
            LLM的输出或None（如果出错）
        """
        attempts = 0
        while attempts < self.max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert in document ranking and evaluation. "
                                                      "Given a question and a set of documents, rate each document based on how well it answers the question or "
                                                      "select the best document that answers the question most comprehensively."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return completion.choices[0].message.content
            except Exception as e:
                attempts += 1
                if attempts < self.max_retries:
                    logger.warning(f"Attempt {attempts} failed. Retrying in {self.retry_delay} seconds... Error: {e}")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed. Error: {e}")
                    return None
    
    def format_relative_ranking_prompt(self, query: str, documents: List[Dict]) -> str:
        """格式化相对排序提示词（选择最佳文档）
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            
        Returns:
            格式化的提示词
        """
        prompt = f"""Given a question and a set of documents, select the best document that answers the question most comprehensively.

Question: {query}

"""
        for i, doc in enumerate(documents):
            content = doc.get("page_content", doc.get("content", ""))
            prompt += f"Document {i+1}:\n{content}\n\n"
        
        prompt += """Your task is to determine which document best answers the question. 
Consider these factors:
1. Relevance: How directly the document addresses the question
2. Completeness: How thoroughly it covers the necessary information
3. Accuracy: Whether the information provided seems correct
4. Clarity: How clearly the information is presented

Respond ONLY with the document number in the format 'Document X' where X is the document number.
If none of the documents answer the question well, specify which is relatively better."""
        
        return prompt
    
    def format_absolute_scoring_prompt(self, query: str, documents: List[Dict]) -> str:
        """格式化绝对评分提示词（给每个文档打分）
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            
        Returns:
            格式化的提示词
        """
        prompt = f"""Rate each document on a scale of 1-5 based on how well it answers the question, where 1 means not relevant at all and 5 means perfectly answering the question with all necessary information.

Question: {query}
"""
        for i, doc in enumerate(documents):
            content = doc.get("page_content", doc.get("content", ""))
            prompt += f"Document {i+1}:\n{content}\n\n"
        
#         prompt += """For each document, provide a rating in the format:
# Document 1: [rating]
# Document 2: [rating]
# ...
# Only provide the ratings in the specified format, no additional explanation."""
        
        return prompt
    
    def parse_relative_ranking(self, response: str) -> Optional[int]:
        """解析相对排名响应，提取最佳文档的索引
        
        Args:
            response: LLM的响应
            
        Returns:
            最佳文档的索引（0-based）或None
        """
        if not response:
            return None
        
        response = response.strip().lower()

        print(response)

        # 尝试多种可能的格式
        patterns = [
            r"document\s+(\d+)",  # Document 1
            r"document\s*:\s*(\d+)",  # Document: 1
            r"best\s*document\s*(?:is|:)?\s*(\d+)",  # Best document is 1
            r"^(\d+)$"  # 只有数字的情况
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    doc_num = int(matches[0])
                    return doc_num - 1  # 转换为0-based索引
                except Exception:
                    continue
        
        # 如果无法匹配，尝试查找文本中的数字
        if "document" in response:
            for word in response.split():
                if word.isdigit():
                    try:
                        return int(word) - 1
                    except Exception:
                        pass
        
        logger.warning(f"Failed to parse ranking from response: {response}")
        return None
    
    def parse_absolute_scores(self, response: str, doc_count: int) -> List[float]:
        """解析绝对评分响应，提取每个文档的评分
        
        Args:
            response: LLM的响应
            doc_count: 文档数量
            
        Returns:
            每个文档的评分列表
        """
        if not response:
            return [0.0] * doc_count

        print(response)

        # 初始化评分为0
        scores = [0.0] * doc_count
        
        # 尝试匹配 "Document X: Y" 格式
        pattern = r"document\s*(\d+)\s*:\s*(\d+(?:\.\d+)?)"
        matches = re.findall(pattern, response.lower())
        
        for match in matches:
            try:
                doc_idx = int(match[0]) - 1  # 转换为0-based索引
                score = float(match[1])
                
                # 确保索引在有效范围内
                if 0 <= doc_idx < doc_count:
                    # 将评分范围统一为0-1之间
                    scores[doc_idx] = min(max(score / 5.0, 0.0), 1.0)
            except Exception as e:
                logger.warning(f"Error parsing score {match}: {e}")
        
        # 如果没有成功解析出任何分数，给出警告
        if all(score == 0.0 for score in scores):
            logger.warning(f"Failed to parse any scores from response: {response}")
        
        return scores
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
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
            logger.warning("No documents to rerank")
            return []

        # 将documents中的数据剪裁到512
        for i, doc in enumerate(documents):
            content = doc.get("page_content", doc.get("content", ""))
            documents[i]["page_content"] = content[:512]

        # 根据评分模式选择不同的提示词
        if self.scoring_mode == "relative":
            prompt = self.format_relative_ranking_prompt(query, documents)
        else:  # absolute
            prompt = self.format_absolute_scoring_prompt(query, documents)
        
        # 调用LLM API
        response = self.call_llm_api(prompt)
        
        # 根据评分模式解析响应
        ranked_docs = []
        if self.scoring_mode == "relative":
            # 解析相对排名
            best_doc_idx = self.parse_relative_ranking(response)
            
            # 根据LLM选择的最佳文档为所有文档分配分数
            for i, doc in enumerate(documents):
                content = doc.get("page_content", doc.get("content", ""))
                metadata = doc.get("metadata", {})
                
                # 为了在排序后保持一定的多样性，我们给第一名1分，其他文档给较低的分数而不是0
                if i == best_doc_idx:
                    score = 1.0
                else:
                    # 其他文档根据与查询的原始顺序给予较低的分数
                    score = 0.1  # 基础分
                
                ranked_docs.append(RankedDocument(
                    page_content=content,
                    score=score,
                    metadata=metadata
                ))
        else:  # absolute
            # 解析绝对评分
            scores = self.parse_absolute_scores(response, len(documents))
            
            # 根据LLM给出的评分为文档分配分数
            for i, doc in enumerate(documents):
                content = doc.get("page_content", doc.get("content", ""))
                metadata = doc.get("metadata", {})
                
                ranked_docs.append(RankedDocument(
                    page_content=content,
                    score=scores[i],
                    metadata=metadata
                ))
        
        # 按分数降序排序
        ranked_docs.sort(key=lambda x: x.score, reverse=True)
        
        # 如果指定了top_k，只返回前top_k个文档
        if top_k is not None and top_k > 0:
            ranked_docs = ranked_docs[:top_k]
        
        return ranked_docs
    
    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[Dict]],
        top_k: Optional[int] = None
    ) -> List[List[RankedDocument]]:
        """批量重排序多个查询的文档
        
        Args:
            queries: 查询文本列表
            documents_list: 每个查询对应的待重排序文档列表
            top_k: 每个查询返回前k个结果
            
        Returns:
            重排序后的文档列表的列表
        """
        results = []
        
        for query, documents in zip(queries, documents_list):
            ranked_docs = self.rerank(query, documents, top_k)
            results.append(ranked_docs)
        
        return results


def enhance_retrieval_with_llm_reranking(
    query: str,
    vector_store,
    k: int = 10,
    top_k: int = 3,
    model_name: str = "Qwen2.5-7B-Instruct-local",
    api_base_url: str = "http://localhost:8000/v1"
) -> List[Dict]:
    """使用LLM重排序增强检索系统
    
    Args:
        query: 用户查询
        vector_store: 向量存储
        k: 初始检索数量
        top_k: 重排序后保留数量
        model_name: LLM模型名称
        api_base_url: API基础URL
        
    Returns:
        重排序后的文档列表
    """
    # 初始检索
    results = vector_store.similarity_search(query, k=k)
    
    # 准备文档列表
    documents = []
    for doc in results:
        documents.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
    
    # 初始化LLM重排序器
    reranker = LLMReranker(
        model_name=model_name,
        api_base_url=api_base_url
    )
    
    # 重排序
    ranked_docs = reranker.rerank(query, documents, top_k=top_k)
    
    # 转换为字典格式
    reranked_results = []
    for doc in ranked_docs:
        reranked_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": doc.score
        })
    
    return reranked_results


if __name__ == "__main__":
    # 测试代码
    from create_faissdb import load_vector_store
    
    # 设置参数
    api_base_url = "http://localhost:8000/v1"
    model_name = "Qwen2.5-7B-Instruct-local"
    faiss_index_path = "database/faiss_new"
    
    # 加载向量存储
    vector_store = load_vector_store(faiss_index_path)
    
    # 测试查询
    test_query = "What should be done if the CDU display does not show the maintenance message during the initial evaluation?"
    
    # 测试增强检索
    results = enhance_retrieval_with_llm_reranking(
        query=test_query,
        vector_store=vector_store,
        k=5,
        top_k=1,
        model_name=model_name,
        api_base_url=api_base_url
    )
    
    # 打印结果
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}: Score = {result['score']}")
        logger.info(f"Content: {result['content'][:200]}...")
        logger.info("-" * 50)
