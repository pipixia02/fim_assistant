from typing import List, Dict, Optional

import nltk
import numpy as np
from langchain_community.vectorstores import FAISS
from loguru import logger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from create_faissdb import load_qa_dataset, load_vector_store

# 配置参数
faiss_index_path = 'database/faiss_new'
test_data_path = 'dataset/retrieval_data/new/test.json'
model_path = 'Models/st_model/best_model_0.690' # 使用训练好的模型
model_path_ori = 'Models/st_model/best_model_0.690'
nltk.data.path.append('/home/hongchang/llm_projects/shenhang/nltk_data')


class HybridRetriever:
    def __init__(self, vector_store: FAISS, use_reranker: bool = False):
        self.stop_words = set(stopwords.words('english'))
        self.vector_store = vector_store
        self.documents = self._prepare_documents()
        self.bm25 = self._initialize_bm25()
        self.use_reranker = use_reranker



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

    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """语义检索，返回带有实际相似度分数的结果"""
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        return [{
            'content': r[0].page_content,
            'metadata': r[0].metadata,
            'score': r[1]  # 使用实际的相似度分数
        } for r in results]

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

    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """
        改进的归一化分数方法
        使用排名进行归一化，避免不同分数分布带来的问题
        """
        if not results:
            return results

        # 提取分数并转换为numpy数组
        scores = np.array([r['score'] for r in results])
        
        # 使用排名归一化
        ranks = np.argsort(np.argsort(-scores))  # 转换为排名（0为最高）
        
        # 将排名转换为0-1范围的分数
        normalized_results = []
        for i, r in enumerate(results):
            # 最高排名得分为1.0
            normalized_score = 1.0 - (ranks[i] / max(1, len(ranks) - 1))
            normalized_results.append(dict(r, normalized_score=normalized_score))
        
        return normalized_results

    def _calculate_alpha(self, query: str) -> float:
        """
        改进的动态alpha计算
        根据查询特性智能调整语义搜索和关键词搜索的权重
        """
        tokens = self._tokenize_text(query)
        
        # 航空领域相关词汇表
        aviation_terms = [
            'maintenance', 'aircraft', 'fault', 'system', 'failure', 
            'indicator', 'procedure', 'valve', 'apu', 'engine', 'light', 
            'warning', 'error', 'check', 'panel', 'switch', 'pressure',
            'hydraulic', 'electrical', 'electronic', 'repair', 'test',
            'troubleshoot', 'inspection', 'component', 'module'
        ]
        
        # 计算航空领域专业词汇比例
        tokens_lower = [t.lower() for t in tokens]
        aviation_term_count = sum(1 for t in tokens_lower if t in aviation_terms)
        aviation_term_ratio = aviation_term_count / max(1, len(tokens))
        
        # 查询长度影响因子
        length_factor = min(0.7, max(0.3, len(tokens) / 10))
        
        # 领域词汇影响因子
        domain_factor = min(0.8, max(0.5, aviation_term_ratio + 0.5))
        
        # 基于查询特征计算最终alpha
        alpha = (length_factor + domain_factor) / 2
        

        
        return alpha

    def _merge_results(self, semantic_results: List[Dict],
                      keyword_results: List[Dict],
                      alpha: Optional[float] = None) -> List[Dict]:
        """
        简化的结果合并方法
        避免过于复杂的计算公式
        """
        # 如果未提供alpha，则动态计算
        if alpha is None:
            query_text = semantic_results[0]['content'] if semantic_results else ""
            alpha = self._calculate_alpha(query_text)

            
        # 归一化分数
        semantic_results = self._normalize_scores(semantic_results)
        keyword_results = self._normalize_scores(keyword_results)
        
        # 创建文档索引
        doc_map = {}
        
        # 处理语义搜索结果
        for res in semantic_results:
            doc_key = res['content']
            doc_map[doc_key] = {
                'content': res['content'],
                'metadata': res['metadata'],
                'semantic_score': res['normalized_score'],
                'keyword_score': 0.0,
                'source': 'semantic'  # 标记来源
            }
        
        # 处理关键词搜索结果
        for res in keyword_results:
            doc_key = res['content']
            if doc_key in doc_map:
                doc_map[doc_key]['keyword_score'] = res['normalized_score']
                # 同时存在于两种结果中
                doc_map[doc_key]['source'] = 'both'
            else:
                doc_map[doc_key] = {
                    'content': res['content'],
                    'metadata': res['metadata'],
                    'semantic_score': 0.0,
                    'keyword_score': res['normalized_score'],
                    'source': 'keyword'  # 标记来源
                }
        
        # 计算最终得分
        final_results = []
        for doc_key, doc in doc_map.items():
            # 简单线性组合，避免过于复杂的公式
            final_score = alpha * doc['semantic_score'] + (1 - alpha) * doc['keyword_score']
            
            # 对同时存在于两种结果中的文档给予额外奖励
            if doc['source'] == 'both':
                final_score *= 1.1  # 10%的提升
            
            final_results.append({
                'content': doc['content'],
                'metadata': doc['metadata'],
                'final_score': final_score,
                'semantic_score': doc['semantic_score'],
                'keyword_score': doc['keyword_score'],
                'source': doc['source']
            })
        
        return sorted(final_results, key=lambda x: x['final_score'], reverse=True)

    def _find_intersection(self, semantic_results: List[Dict], keyword_results: List[Dict], top_k: int) -> List[Dict]:
        """
        找出同时出现在两种检索结果中的项，这些项很可能是最相关的
        
        Args:
            semantic_results: 语义搜索结果
            keyword_results: 关键词搜索结果
            top_k: 需要返回的结果数量
            
        Returns:
            同时出现在两种检索结果中的文档，不足则由语义搜索结果补充
        """
        # 记录初始结果数量
        # logger.info(f"语义搜索结果数量: {len(semantic_results)}")
        # logger.info(f"关键词搜索结果数量: {len(keyword_results)}")

        # 为了快速查找，创建一个从内容到结果的映射
        semantic_contents = {r['content']: r for r in semantic_results}
        keyword_contents = {r['content']: r for r in keyword_results}
        
        # 找出交集
        intersection = []
        for content in set(semantic_contents.keys()) & set(keyword_contents.keys()):
            sr = semantic_contents[content]
            kr = keyword_contents[content]
            
            # 计算组合分数 - 使用排名作为权重基础
            s_rank = [i for i, r in enumerate(semantic_results) if r['content'] == content][0] + 1
            k_rank = [i for i, r in enumerate(keyword_results) if r['content'] == content][0] + 1
            
            # 转换为排名得分 (1/rank)，排名越高得分越高
            s_rank_score = 1.0 / s_rank
            k_rank_score = 1.0 / k_rank
            
            # 基于两种排名的加权平均，语义搜索权重略高
            combined_score = (s_rank_score * 0.6 + k_rank_score * 0.4)
            
            intersection.append({
                'content': content,
                'metadata': sr['metadata'],
                'final_score': combined_score,
                'semantic_score': sr.get('normalized_score', sr['score']),
                'keyword_score': kr.get('normalized_score', kr['score']),
                'semantic_rank': s_rank,
                'keyword_rank': k_rank,
                'source': 'intersection'
            })
        
        # 按组合得分排序
        intersection = sorted(intersection, key=lambda x: x['final_score'], reverse=True)
        
        # 记录交集数量
        # logger.info(f"找到的交集文档数量: {len(intersection)}")
        
        # 如果交集不足，从语义结果中补充
        supplement_count = 0
        if len(intersection) < top_k:
            intersection_contents = {r['content'] for r in intersection}
            
            # 按原始顺序添加语义结果中未在交集中的文档
            for i, sr in enumerate(semantic_results):
                if sr['content'] not in intersection_contents:
                    # 添加一些额外信息以便区分
                    intersection.append({
                        'content': sr['content'],
                        'metadata': sr['metadata'],
                        'final_score': sr.get('normalized_score', sr['score']) * 0.7,  # 降低权重，确保交集结果排在前面
                        'semantic_score': sr.get('normalized_score', sr['score']),
                        'keyword_score': 0,
                        'semantic_rank': i + 1,
                        'keyword_rank': None,
                        'source': 'semantic_only'
                    })
                    
                    supplement_count += 1
                    if len(intersection) >= top_k:
                        break
        
        # 记录补充数量
        # logger.info(f"从语义搜索中补充的文档数量: {supplement_count}")
        
        # 最终按得分重新排序
        result = sorted(intersection, key=lambda x: x['final_score'], reverse=True)[:top_k]
        
        # 记录返回的结果类型
        intersection_count = sum(1 for r in result if r['source'] == 'intersection')
        semantic_only_count = sum(1 for r in result if r['source'] == 'semantic_only')
        
        # logger.info(f"最终返回结果: 交集文档 {intersection_count} 个, 仅语义文档 {semantic_only_count} 个")
        
        # 如果没有找到任何结果（极端情况），返回语义搜索结果
        if not result:
            logger.warning("没有找到任何交集或补充结果，回退到语义搜索结果")
            return [
                {
                    'content': sr['content'],
                    'metadata': sr['metadata'],
                    'final_score': sr.get('normalized_score', sr['score']),
                    'semantic_score': sr.get('normalized_score', sr['score']),
                    'keyword_score': 0,
                    'semantic_rank': i + 1,
                    'keyword_rank': None,
                    'source': 'semantic_fallback'
                } for i, sr in enumerate(semantic_results[:top_k])
            ]
        
        return result

    def _rerank_results(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        """使用交叉编码器重排序结果"""
        if not self.use_reranker or not hasattr(self, 'cross_encoder'):
            return results[:top_k]

        try:
            # 准备输入
            pairs = [(query, result['content']) for result in results]
            # 计算分数
            scores = self.cross_encoder.predict(pairs)
            # 更新结果
            for i, result in enumerate(results):
                result['rerank_score'] = scores[i]
            # 重新排序
            reranked_results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)

            return reranked_results[:top_k]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:top_k]

    def search(self, query: str, top_k: int = 5, alpha: Optional[float] = None, use_reranker: bool = None, use_intersection: bool = True, return_format: str = "text"):
        """
        改进的混合检索方法，增加了交集模式
        
        Args:
            query: 搜索查询
            top_k: 返回的结果数量
            alpha: 语义搜索的权重，None表示使用动态计算
            use_reranker: 是否使用重排序
            use_intersection: 是否使用交集模式
            return_format: 返回格式，"text"表示返回格式化文本，"list"表示返回结构化列表
            
        Returns:
            格式化的检索结果文本或结构化的结果列表
        """
        # 确定是否使用重排序
        use_reranker = self.use_reranker if use_reranker is None else use_reranker
        
        # 保证检索足够多的候选文档
        search_top_k = max(30, top_k * 5)  # 更大的候选池，确保有足够的交集
        
        # 获取两种检索结果
        semantic_results = self._semantic_search(query, search_top_k)
        keyword_results = self._keyword_search(query, search_top_k)
        
        # 根据不同策略选择结果
        if use_intersection:
            # 使用交集策略
            logger.info("Using intersection strategy")
            merged_results = self._find_intersection(semantic_results, keyword_results, top_k)
        elif alpha == 1.0:
            logger.info("Using semantic search only")
            merged_results = self._normalize_scores(semantic_results)
        elif alpha == 0.0:
            logger.info("Using keyword search only")
            merged_results = self._normalize_scores(keyword_results)
        else:
            # 使用标准混合策略
            logger.info("Using standard hybrid strategy")
            merged_results = self._merge_results(semantic_results, keyword_results, alpha)
        
        # 如果启用重排序，则进行重排序
        if use_reranker:
            logger.info("Applying reranking")
            final_results = self._rerank_results(query, merged_results, top_k)
        else:
            final_results = merged_results[:top_k]
            
        # 记录检索结果的分数信息
        for i, result in enumerate(final_results[:top_k]):
            source_info = f", source={result.get('source', 'unknown')}"
            rank_info = ""
            if 'semantic_rank' in result and 'keyword_rank' in result:
                rank_info = f", semantic_rank={result.get('semantic_rank', 'N/A')}, keyword_rank={result.get('keyword_rank', 'N/A')}"
                
            logger.debug(f"Result {i + 1}: final_score={result.get('final_score', 0):.4f}, "
                        f"semantic_score={result.get('semantic_score', 0):.4f}, "
                        f"keyword_score={result.get('keyword_score', 0):.4f}"
                        f"{source_info}{rank_info}")
        
        # 根据返回格式选择返回结果
        if return_format == "list":
            # 返回结构化的结果列表
            return final_results
        else:
            # 格式化输出文本
            info_text = ''
            for result in final_results[:top_k]:
                info_text += f'''
###description of question:###
{result['content']}
###Fault Isolation:###
{result['metadata'].get('original_procedure', 'N/A')}\n'''
            
            return info_text




def evaluate_hybrid_retrieval(vector_store, qa_pairs: List[Dict], top_k: int, alpha: Optional[float] = 0.7, 
                              use_reranker: bool = False, use_intersection: bool = False):
    """
    评估混合检索的性能

    Args:
        vector_store: FAISS向量存储
        qa_pairs: 问答对列表
        top_k: 返回的top k个结果
        alpha: 语义检索权重（1-alpha为关键词检索权重），None表示使用动态alpha
        use_reranker: 是否使用重排序
        use_intersection: 是否使用交集搜索策略
    """
    # 初始化混合检索器
    hybrid_retriever = HybridRetriever(vector_store, use_reranker=use_reranker)
    total_questions = len(qa_pairs)
    correct_retrievals = 0
    all_precision = []
    all_recall = []
    all_f1 = []

    # 记录交集搜索统计
    intersection_stats = {
        'total_intersections': 0,
        'avg_intersection_size': 0,
        'intersection_correct': 0
    }

    for qa_pair in qa_pairs:
        question = qa_pair['question']

        # 获取两种检索结果
        semantic_results = hybrid_retriever._semantic_search(question, top_k * 3)
        keyword_results = hybrid_retriever._keyword_search(question, top_k * 3)
        
        # 根据指定策略选择结果
        if use_intersection:
            # 使用交集策略
            merged_results = hybrid_retriever._find_intersection(semantic_results, keyword_results, top_k)
            
            # 收集交集统计信息
            intersection_items = [r for r in merged_results if r.get('source') == 'intersection']
            intersection_stats['total_intersections'] += len(intersection_items)
            
            # 检查交集中是否包含正确答案
            intersection_correct = any(
                r['metadata'].get('question', '') == question 
                for r in intersection_items
            )
            if intersection_correct:
                intersection_stats['intersection_correct'] += 1
                
        elif alpha == 1.0:
            # 仅使用语义搜索
            merged_results = semantic_results[:top_k]
        elif alpha == 0.0:
            # 仅使用关键词搜索
            merged_results = keyword_results[:top_k]
        else:
            # 使用标准混合策略
            merged_results = hybrid_retriever._merge_results(semantic_results, keyword_results, alpha=alpha)

        # 如果启用重排序，则进行重排序
        if use_reranker:
            final_results = hybrid_retriever._rerank_results(question, merged_results, top_k)
        else:
            final_results = merged_results[:top_k]

        # 提取检索到的问题
        retrieved_questions = [
            doc['metadata'].get('question', '')
            for doc in final_results[:top_k]
        ]

        # 计算正确检索数量
        true_positives = sum(1 for q in retrieved_questions if q == question)

        # 计算精确度和召回率
        precision = true_positives / len(retrieved_questions) if retrieved_questions else 0
        recall = 1 if true_positives > 0 else 0  # 每个问题只有一个正确答案
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        
        if true_positives > 0:
            correct_retrievals += 1

    # 计算评估指标
    metrics = {
        'accuracy': correct_retrievals / total_questions,
        'precision': np.mean(all_precision),
        'recall': np.mean(all_recall),
        'f1_score': np.mean(all_f1),
        'total_questions': total_questions,
        'correct_retrievals': correct_retrievals
    }
    
    # 添加交集统计信息
    if use_intersection and intersection_stats['total_intersections'] > 0:
        metrics['intersection_stats'] = {
            'total_intersections': intersection_stats['total_intersections'],
            'avg_intersection_size': intersection_stats['total_intersections'] / total_questions,
            'intersection_accuracy': intersection_stats['intersection_correct'] / total_questions
        }

    return metrics


def main():
    """主函数，用于测试和评估混合检索"""
    # 加载测试数据集
    qa_pairs = load_qa_dataset(test_data_path)
    if not qa_pairs:
        logger.error(f"Failed to load QA dataset from {test_data_path}")
        return

    # 加载向量存储
    vector_store = load_vector_store(faiss_index_path)
    if vector_store is None:
        logger.error(f"Failed to load vector store from {faiss_index_path}")
        return

    # 评估各种检索策略
    # 1. 原始检索模型
    original_metrics = evaluate_hybrid_retrieval(vector_store, qa_pairs, top_k=5, alpha=1.0)
    logger.info(f"ORIGINAL Performance:\nAccuracy: {original_metrics['accuracy']:.4f}\n"
                f"Precision: {original_metrics['precision']:.4f}\n"
                f"Recall: {original_metrics['recall']:.4f}\n"
                f"F1 Score: {original_metrics['f1_score']:.4f}")

    # 2. 仅使用BM25
    sparse_metrics = evaluate_hybrid_retrieval(vector_store, qa_pairs, top_k=5, alpha=0.0)
    logger.info(f"SPARSE_BM25 Performance:\nAccuracy: {sparse_metrics['accuracy']:.4f}\n"
                f"Precision: {sparse_metrics['precision']:.4f}\n"
                f"Recall: {sparse_metrics['recall']:.4f}\n"
                f"F1 Score: {sparse_metrics['f1_score']:.4f}")

    # 3. 混合检索 alpha=0.3
    hybrid_metrics_0_3 = evaluate_hybrid_retrieval(vector_store, qa_pairs, top_k=5, alpha=0.3)
    logger.info(f"HYBRID_ALPHA_0.3 Performance:\nAccuracy: {hybrid_metrics_0_3['accuracy']:.4f}\n"
                f"Precision: {hybrid_metrics_0_3['precision']:.4f}\n"
                f"Recall: {hybrid_metrics_0_3['recall']:.4f}\n"
                f"F1 Score: {hybrid_metrics_0_3['f1_score']:.4f}")

    # 4. 混合检索 alpha=0.7
    hybrid_metrics_0_7 = evaluate_hybrid_retrieval(vector_store, qa_pairs, top_k=5, alpha=0.7)
    logger.info(f"HYBRID_ALPHA_0.7 Performance:\nAccuracy: {hybrid_metrics_0_7['accuracy']:.4f}\n"
                f"Precision: {hybrid_metrics_0_7['precision']:.4f}\n"
                f"Recall: {hybrid_metrics_0_7['recall']:.4f}\n"
                f"F1 Score: {hybrid_metrics_0_7['f1_score']:.4f}")
                
    # 5. 交集搜索
    intersection_metrics = evaluate_hybrid_retrieval(vector_store, qa_pairs, top_k=5, use_intersection=True)
    logger.info(f"INTERSECTION Performance:\nAccuracy: {intersection_metrics['accuracy']:.4f}\n"
                f"Precision: {intersection_metrics['precision']:.4f}\n"
                f"Recall: {intersection_metrics['recall']:.4f}\n"
                f"F1 Score: {intersection_metrics['f1_score']:.4f}")
    



if __name__ == '__main__':
    main()