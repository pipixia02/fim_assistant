from typing import List, Dict, Set
from loguru import logger
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re


class ImprovedRetriever:
    def __init__(self, vector_store):
        """改进的检索器"""
        self.vector_store = vector_store
        self.stop_words = set(stopwords.words('english'))

        # 1. 技术术语库
        self.technical_terms = {
            'maintenance': {'repair', 'check', 'inspect', 'replace', 'test'},
            'system': {'component', 'unit', 'assembly', 'equipment'},
            'fault': {'failure', 'malfunction', 'defect', 'error'},
            'procedure': {'step', 'instruction', 'operation', 'task'},
            # 可以扩充更多术语
        }

        # 2. 文档结构模式
        self.doc_patterns = {
            'numbered_step': r'\([0-9]+\)',  # 匹配 (1), (2) 等
            'reference_code': r'[A-Z0-9]+-[A-Z0-9]+',  # 匹配参考代码
            'manual_section': r'[A-Z]{3}\s+ALL'  # 匹配手册章节
        }

        # 3. 初始化BM25
        self.documents = self._prepare_documents()
        self.bm25 = self._initialize_bm25()

        logger.info("Initialized Improved Retriever")

    def search(self, query: str, k: int = 3) -> List[Document]:
        """改进的检索方法"""
        try:
            # 1. 向量检索
            dense_results = self.vector_store.similarity_search(query, k=k * 2)

            # 2. BM25检索
            sparse_results = self._keyword_search(query, k=k * 2)

            # 3. 合并结果
            combined_results = self._merge_results(dense_results, sparse_results)

            # 4. 技术术语验证和结构验证
            verified_results = self._verify_documents(combined_results, query)

            # 5. 最终排序
            final_results = self._rank_results(verified_results, query)[:k]

            return final_results

        except Exception as e:
            logger.error(f"Error in search: {e}")
            return self.vector_store.similarity_search(query, k=k)

    def _verify_documents(self, docs: List[Document], query: str) -> List[Dict]:
        """验证文档的技术术语和结构"""
        verified_docs = []
        query_terms = self._extract_technical_terms(query)

        for doc in docs:
            score = 0
            content = doc.page_content

            # 1. 技术术语匹配分数
            doc_terms = self._extract_technical_terms(content)
            term_score = len(query_terms & doc_terms)

            # 2. 文档结构分数
            structure_score = self._check_document_structure(content)

            # 3. 上下文连贯性分数
            context_score = self._check_context_coherence(content)

            # 计算总分
            total_score = term_score + structure_score + context_score

            verified_docs.append({
                'document': doc,
                'score': total_score,
                'terms': doc_terms,
                'structure_score': structure_score
            })

        return verified_docs

    def _extract_technical_terms(self, text: str) -> Set[str]:
        """提取技术术语"""
        terms = set()
        words = word_tokenize(text.lower())

        # 1. 直接匹配
        for word in words:
            if word in self.technical_terms:
                terms.add(word)
                terms.update(self.technical_terms[word])

        # 2. 匹配复合术语
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i + 1]}"
            if bigram in self.technical_terms:
                terms.add(bigram)

        return terms

    def _check_document_structure(self, content: str) -> float:
        """检查文档结构"""
        score = 0

        # 1. 检查编号步骤
        if re.search(self.doc_patterns['numbered_step'], content):
            score += 1

        # 2. 检查参考代码
        if re.search(self.doc_patterns['reference_code'], content):
            score += 0.5

        # 3. 检查手册章节
        if re.search(self.doc_patterns['manual_section'], content):
            score += 0.5

        return score

    def _check_context_coherence(self, content: str) -> float:
        """检查上下文连贯性"""
        score = 0

        # 1. 检查步骤顺序
        steps = re.findall(r'\([0-9]+\)', content)
        if steps:
            # 检查步骤是否连续
            step_nums = [int(re.search(r'\d+', step).group()) for step in steps]
            if sorted(step_nums) == step_nums:
                score += 1

        # 2. 检查章节引用
        if re.search(r'refer to|see|reference', content.lower()):
            score += 0.5

        return score

    def _rank_results(self, verified_docs: List[Dict], query: str) -> List[Document]:
        """最终排序"""
        # 1. 计算查询相关性
        query_terms = self._extract_technical_terms(query)

        for doc in verified_docs:
            # 技术术语匹配权重
            term_weight = len(doc['terms'] & query_terms) * 0.4
            # 文档结构权重
            structure_weight = doc['structure_score'] * 0.3
            # 原始分数权重
            original_weight = doc['score'] * 0.3

            # 最终分数
            doc['final_score'] = term_weight + structure_weight + original_weight

        # 2. 排序
        verified_docs.sort(key=lambda x: x['final_score'], reverse=True)

        return [doc['document'] for doc in verified_docs]

    def _merge_results(self, dense_results: List[Document],
                       sparse_results: List[Dict]) -> List[Document]:
        """合并检索结果"""
        seen = set()
        merged = []

        # 1. 添加密集检索结果
        for doc in dense_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)

        # 2. 添加稀疏检索结果
        for doc in sparse_results:
            if doc['content'] not in seen:
                seen.add(doc['content'])
                merged.append(Document(
                    page_content=doc['content'],
                    metadata=doc['metadata']
                ))

        return merged