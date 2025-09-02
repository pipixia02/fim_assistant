from typing import List, Dict, Optional, Union, Tuple
import nltk
import numpy as np
from langchain_community.vectorstores import FAISS
from loguru import logger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from create_faissdb import load_qa_dataset, load_vector_store

# 配置参数
nltk.data.path.append('/home/hongchang/llm_projects/shenhang/nltk_data')

class MultiHopRetriever:
    def __init__(self, vector_store: FAISS, use_reranker: bool = False, hops: int = 2, use_intersection: bool = True):
        """
        初始化多跳检索器
        
        Args:
            vector_store: FAISS向量存储
            use_reranker: 是否使用重排序
            hops: 检索跳数，默认为2跳
            use_intersection: 是否使用交集检索
        """
        self.stop_words = set(stopwords.words('english'))
        self.vector_store = vector_store
        self.documents = self._prepare_documents()
        self.bm25 = self._initialize_bm25()
        self.use_reranker = use_reranker
        self.hops = hops
        self.use_intersection = use_intersection
        
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
        """对文本进行分词，去除停用词"""
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token not in self.stop_words]
        
    def search(self, query: str, top_k: int = 5, method: str = "frequency") -> List:
        """
        执行多跳检索
        
        Args:
            query: 查询文本
            top_k: 返回的结果数量
            method: 查询增强方法，可选择：
                - frequency: 基于词频的增强
                - entity_extraction: 基于实体提取的增强
                
        Returns:
            检索结果
        """
        logger.info(f"开始执行多跳检索，查询: {query[:50]}...")
        logger.info(f"使用交集检索: {self.use_intersection}, 方法: {method}")
        
        # 初始化结果集
        all_results = []
        current_query = query
        
        # 首跳检索
        logger.info(f"执行首跳检索，查询: {current_query[:50]}...")
        
        if self.use_intersection:
            # 使用交集检索
            try:
                from retrieval.hybrid_retrieval import HybridRetriever
                hybrid_retriever = HybridRetriever(self.vector_store, use_reranker=self.use_reranker)
                first_hop_results = hybrid_retriever.search(
                    query, 
                    top_k=top_k*2, 
                    use_intersection=True,
                    return_format="list"
                )
                
                logger.info(f"首跳交集检索结果数量: {len(first_hop_results)}")
                all_results.extend(first_hop_results)
            except Exception as e:
                logger.error(f"首跳交集检索出错: {e}")
                # 回退到向量检索
                try:
                    first_hop_results = self.vector_store.similarity_search(query, k=top_k*2)
                    first_hop_results = [
                        {'content': doc.page_content, 'metadata': doc.metadata} 
                        for doc in first_hop_results
                    ]
                    logger.info(f"回退到向量检索，结果数量: {len(first_hop_results)}")
                    all_results.extend(first_hop_results)
                except Exception as e2:
                    logger.error(f"回退向量检索也失败: {e2}")
                    return []
        else:
            # 使用纯向量检索
            try:
                first_hop_results = self.vector_store.similarity_search(query, k=top_k*2)
                first_hop_results = [
                    {'content': doc.page_content, 'metadata': doc.metadata} 
                    for doc in first_hop_results
                ]
                logger.info(f"首跳向量检索结果数量: {len(first_hop_results)}")
                all_results.extend(first_hop_results)
            except Exception as e:
                logger.error(f"首跳向量检索出错: {e}")
                return []
        
        # 如果只有一跳，直接返回结果
        if self.hops <= 1:
            return all_results[:top_k]
        
        # 后续跳使用增强查询
        for hop in range(1, self.hops):
            try:
                # 根据前一跳结果增强查询
                current_query = self._enhance_query(query, all_results, method=method)
                logger.info(f"增强后的查询: {current_query[:50]}...")
                
                # 使用增强查询进行下一跳的检索
                logger.info(f"执行第 {hop+1} 跳检索，查询: {current_query[:50]}...")
                
                if self.use_intersection:
                    # 使用交集检索
                    try:
                        from retrieval.hybrid_retrieval import HybridRetriever
                        hybrid_retriever = HybridRetriever(self.vector_store, use_reranker=self.use_reranker)
                        hop_results = hybrid_retriever.search(
                            current_query, 
                            top_k=top_k, 
                            use_intersection=True,
                            return_format="list"
                        )
                        
                        logger.info(f"第 {hop+1} 跳交集检索结果数量: {len(hop_results)}")
                        all_results.extend(hop_results)
                    except Exception as e:
                        logger.error(f"第 {hop+1} 跳交集检索出错: {e}")
                        # 回退到向量检索
                        hop_results = self.vector_store.similarity_search(current_query, k=top_k)
                        hop_results = [
                            {'content': doc.page_content, 'metadata': doc.metadata} 
                            for doc in hop_results
                        ]
                        logger.info(f"回退到向量检索，结果数量: {len(hop_results)}")
                        all_results.extend(hop_results)
                else:
                    # 使用纯向量检索
                    hop_results = self.vector_store.similarity_search(current_query, k=top_k)
                    hop_results = [
                        {'content': doc.page_content, 'metadata': doc.metadata} 
                        for doc in hop_results
                    ]
                    logger.info(f"第 {hop+1} 跳向量检索结果数量: {len(hop_results)}")
                    all_results.extend(hop_results)
            except Exception as e:
                logger.error(f"第 {hop+1} 跳检索出错: {e}")
                break
        
        # 合并去重
        try:
            merged_results = self._merge_hop_results(all_results)
            logger.info(f"合并后的结果数量: {len(merged_results)}")
        except Exception as e:
            logger.error(f"合并结果出错: {e}")
            merged_results = all_results
        
        # 返回top_k个结果
        return merged_results[:top_k]
        
    def _enhance_query(self, original_query: str, results: List, method: str = "frequency") -> str:
        """
        基于当前跳的结果增强原始查询，提供多种增强方法
        
        Args:
            original_query: 原始查询
            results: 检索结果
            method: 增强方法，可选择：
                    - frequency: 基于词频统计的增强（默认）
                    - entity_extraction: 基于实体提取的增强
                    
        Returns:
            增强后的查询
        """
        # 提取前3个结果的内容
        context = ""
        for doc in results[:3]:
            print(doc)
            # 处理不同的文档格式
            if hasattr(doc, 'page_content'):
                # Document对象 (向量检索结果)
                content = doc.page_content
            elif isinstance(doc, dict):
                # 字典对象 (混合检索结果)
                content = doc.get('content', '')
            elif isinstance(doc, str):
                # 字符串对象
                content = doc
            else:
                # 其他类型，尝试获取内容
                try:
                    content = doc.get('content', '')
                except:
                    content = str(doc)
            
            # 添加到上下文
            context += content + " "
        
        # 根据不同方法增强查询
        if method == "frequency":
            # 基于词频的增强方法
            enhanced_query = self._enhance_by_frequency(original_query, context)
        elif method == "entity_extraction":
            # 基于实体提取的增强方法
            enhanced_query = self._enhance_by_entity_extraction(original_query, context)
        else:
            # 默认使用词频方法
            enhanced_query = self._enhance_by_frequency(original_query, context)
        
        logger.info(f"增强查询({method}): {enhanced_query[:100]}...")
        
        return enhanced_query
        
    def _enhance_by_frequency(self, original_query: str, context: str) -> str:
        """基于词频的查询增强方法"""
        # 分词并去除停用词
        tokens = self._tokenize_text(context)
        
        # 统计词频
        from collections import Counter
        word_counts = Counter(tokens)
        
        # 获取最常见的10个词
        most_common = [word for word, _ in word_counts.most_common(10)]
        
        # 增强查询
        enhanced_query = original_query + " " + " ".join(most_common)
        
        return enhanced_query
        
    def _enhance_by_entity_extraction(self, original_query: str, context: str) -> str:
        """基于实体提取的查询增强方法"""
        try:
            import spacy
            # 尝试加载spaCy模型，如果不存在则使用简单规则
            try:
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(context)
                # 提取实体
                entities = set()
                for ent in doc.ents:
                    # 只添加有意义的实体类型
                    if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                        entities.add(ent.text)
                
                # 增强查询
                entity_text = " ".join(list(entities)[:7])  # 限制实体数量
                enhanced_query = original_query + " " + entity_text
                
                return enhanced_query
            except:
                # 简单规则：使用大写开头的词作为实体
                import re
                potential_entities = re.findall(r'\b[A-Z][a-zA-Z]+\b', context)
                entities = list(set(potential_entities))[:7]  # 限制实体数量
                enhanced_query = original_query + " " + " ".join(entities)
                
                return enhanced_query
        except ImportError:
            # 如果spaCy未安装，回退到词频方法
            logger.warning("spaCy未安装，回退到词频方法")
            return self._enhance_by_frequency(original_query, context)
        
    def _merge_hop_results(self, results: List) -> List:
        """
        合并多跳检索的结果，去除重复项
        
        Args:
            results: 所有跳的检索结果
            
        Returns:
            去重后的结果列表
        """
        # 使用文档内容或ID作为唯一标识符
        unique_results = {}
        
        for res in results:
            # 处理不同的文档格式
            if hasattr(res, 'page_content'):
                # Document对象 (向量检索结果)
                content = res.page_content
                doc_id = res.metadata.get('id', content[:50])
                if doc_id not in unique_results:
                    unique_results[doc_id] = res
            elif isinstance(res, dict):
                # 字典对象 (混合检索结果)
                content = res.get('content', '')
                doc_id = res.get('metadata', {}).get('id', content[:50])
                if doc_id not in unique_results:
                    unique_results[doc_id] = res
            elif isinstance(res, str):
                # 字符串对象
                content = res
                doc_id = content[:50]  # 使用内容前50个字符作为ID
                if doc_id not in unique_results:
                    unique_results[doc_id] = res
            else:
                # 其他类型，尝试获取内容
                try:
                    content = res.get('content', '')
                except:
                    content = str(res)
                doc_id = content[:50]
                if doc_id not in unique_results:
                    unique_results[doc_id] = res
        
        # 转回列表
        return list(unique_results.values())


def search_info(question: str, vector_store: FAISS, top_k: int = 5, 
                use_reranker: bool = False, hops: int = 2, method: str = "frequency",
                use_intersection: bool = False) -> List:
    """
    搜索信息的入口函数
    
    Args:
        question: 查询问题
        vector_store: FAISS向量存储
        top_k: 返回的结果数量
        use_reranker: 是否使用重排序器
        hops: 检索跳数
        method: 查询增强方法
        use_intersection: 是否使用交集检索
        
    Returns:
        检索结果列表
    """
    try:
        # 初始化多跳检索器
        retriever = MultiHopRetriever(
            vector_store=vector_store, 
            use_reranker=use_reranker, 
            hops=hops,
            use_intersection=use_intersection
        )
        
        # 执行多跳检索
        results = retriever.search(question, top_k=top_k, method=method)
        return results
    except Exception as e:
        logger.error(f"多跳检索出错: {e}")
        # 如果多跳检索失败，回退到基本向量检索
        try:
            results = vector_store.similarity_search(question, k=top_k)
            return [
                {'content': doc.page_content, 'metadata': doc.metadata} 
                for doc in results
            ]
        except Exception as e2:
            logger.error(f"基本向量检索也失败: {e2}")
            return []
