import json
import random
from typing import List, Dict, Optional
import numpy as np
from loguru import logger
from tqdm import tqdm
from create_faissdb import load_qa_dataset, load_vector_store
from retrieval.hybrid_retrieval import HybridRetriever
from retrieval.llm_reranker import LLMReranker

# key = 'sk-JJWo7wRovwXbgVl8gs2q1F4fns8sqOuPecxybkmCY0HEaOib'
key = 'sk-P8CCBqMp5sAqZ5asHW7GnLUEmW4FPwbT9lOuaZqQ28wr7rGK'
# 配置参数
faiss_index_path = 'database/faiss_prompt1_sum'
model_path = 'Models/retrieval_models/prompt1_model/sum_model_0.759'

# test_data_path = 'dataset/retrieval_data/prompt1/summarise/test.json'
test_data_path = 'dataset/retrieval_data/prompt1/summarise/test.json'
qa_data_path = 'dataset/qa_data/qa_dataset_merged.json'

hard_data_lora = 'dataset/lora_data/prompt1_top5.json'
hard_data_vector = 'dataset/retrieval_data/prompt1/summarise/hard_vector_data.json'


# 设置更合理的top_k值
top_k = 10  # 增加top_k值以提高召回率

def evaluate_retrieval(vector_store, qa_pairs: List[Dict], top_k):
    total_questions = len(qa_pairs)
    correct_retrievals = 0
    all_precision = []
    all_recall = []
    all_f1 = []
    logger.info(f"Evaluating basic retrieval with top_k={top_k}...")
    
    # 用于向量检索微调的数据
    hard_data = []
    
    # 用于LLM重排器的数据 - 相对评分模式和绝对评分模式
    reranker_relative_data = []  # 相对评分模式（选择最佳文档）
    reranker_absolute_data = []  # 绝对评分模式（为每个文档打分）
    
    for qa_pair in qa_pairs:
        question = qa_pair['question']
        correct_answer = qa_pair['relevant_doc']
        
        # 获取向量检索结果
        results = vector_store.similarity_search(question, k=top_k)
        vector_docs = [res.page_content for res in results]
        
        # 合并结果并去重
        all_candidates = list(set(vector_docs))
        
        # 检查正确答案是否在候选集中
        if len(all_candidates) > top_k:
            all_candidates = all_candidates[:top_k]
            
        # 确保正确答案在候选集中
        correct_in_candidates = correct_answer in all_candidates
        if not correct_in_candidates:
            # 如果不在，替换一个随机候选
            random_idx = random.randint(0, len(all_candidates) - 1)
            all_candidates[random_idx] = correct_answer
        
        # 找出正确答案的索引
        correct_idx = all_candidates.index(correct_answer)
        
        # 创建向量检索微调数据
        for res in results:
            if question == res.metadata.get('question', ''):
                continue
            hard_data.append({
                "question": question,
                "relevant_doc": qa_pair['relevant_doc'],
                "irrelevant_doc": res.page_content,
            })
        
        # 为LLM重排器创建相对评分模式的数据
        # 格式: 问题 + 候选文档列表 + 正确答案索引
        relative_sample = {
            "query": question,
            "documents": [{"page_content": doc} for doc in all_candidates],
            "best_doc_idx": correct_idx  # 0-based索引，与LLM重排器使用的一致
        }
        reranker_relative_data.append(relative_sample)
        
        # 为LLM重排器创建绝对评分模式的数据
        # 格式: 问题 + 候选文档列表 + 每个文档的得分
        # 这里我们给正确答案5分，其他文档根据与问题的相似度给1-3分
        doc_scores = []
        for i, doc in enumerate(all_candidates):
            if i == correct_idx:
                score = 5.0  # 正确答案得满分
            else:
                # 这里可以用简单的启发式方法给分，或者使用TF-IDF等计算相似度
                # 这里简单起见，随机给1-3分
                score = random.uniform(1.0, 3.0)
            doc_scores.append(score)
        
        absolute_sample = {
            "query": question,
            "documents": [{"page_content": doc} for doc in all_candidates],
            "scores": doc_scores  # 每个文档的分数，对应索引
        }
        reranker_absolute_data.append(absolute_sample)
        
        # 计算评估指标
        retrieved_questions = [doc.metadata.get('question', '') for doc in results]
        true_positives = sum(1 for q in retrieved_questions if q == question)
        precision = true_positives / len(retrieved_questions) if retrieved_questions else 0
        recall = 1 if true_positives > 0 else 0  # Since we only have one relevant document per question
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        
        if true_positives > 0:
            correct_retrievals += 1
    
    # 保存向量检索微调数据
    # with open('./dataset/retrieval_data/prompt1/hard_vector_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(hard_data, f, ensure_ascii=False, indent=2)
    # print(f'保存向量检索微调数据成功，共{len(hard_data)}条')
    #
    # # 保存LLM重排器相对评分模式数据
    # with open('./dataset/llm_reranker/relative_ranking_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(reranker_relative_data, f, ensure_ascii=False, indent=2)
    # print(f'保存LLM重排器相对评分数据成功，共{len(reranker_relative_data)}条')
    #
    # # 保存LLM重排器绝对评分模式数据
    # with open('./dataset/llm_reranker/absolute_scoring_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(reranker_absolute_data, f, ensure_ascii=False, indent=2)
    # print(f'保存LLM重排器绝对评分数据成功，共{len(reranker_absolute_data)}条')
    
    # Calculate overall metrics
    accuracy = correct_retrievals / total_questions
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)
    
    # 计算平均指标
    return {
        'accuracy': accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1,
        'total_questions': total_questions,
        'correct_retrievals': correct_retrievals
    }

def evaluate_retrieval_strategy(vector_store, qa_pairs: List[Dict], top_k: int, strategy: str = "sparse"):
    """
    通用评估不同检索策略的函数
    
    Args:
        vector_store: 向量存储
        qa_pairs: 问答对列表
        top_k: 返回的top k个结果
        strategy: 检索策略，可选值包括：
                  - "sparse": 仅使用BM25稀疏检索
                  - "simple_hybrid": 使用简易混合检索（向量+BM25，结果合并去重）
                  - "combined_topk": 组合检索（向量+BM25，合并各自的top-k结果）
                  
    Returns:
        Dict: 评估指标
    """
    # 初始化混合检索器
    retriever = HybridRetriever(vector_store, use_reranker=False)
    
    total_questions = len(qa_pairs)
    correct_retrievals = 0
    all_precision = []
    all_recall = []
    all_f1 = []
    all_mrr = []  # 平均倒数排名（只在部分策略中计算）
    
    strategy_display_name = {
        "sparse": "sparse retrieval (BM25)",
        "simple_hybrid": "simple hybrid retrieval",
        "combined_topk": "combined top-k retrieval"
    }.get(strategy, strategy)
    
    logger.info(f"Evaluating {strategy_display_name} with top_k={top_k}...")
    
    for qa_pair in tqdm(qa_pairs, desc=f"Evaluating {strategy}"):
        question = qa_pair['question']

        # 根据策略获取检索结果
        if strategy == "sparse":
            # 只使用关键词检索
            keyword_results = retriever._keyword_search(question, top_k=top_k)
            merged_results = [
                {
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'source': 'sparse'
                }
                for doc in keyword_results
            ]
        else:
            # 执行向量检索
            vector_results = vector_store.similarity_search(question, k=top_k)
            # 执行稀疏检索(BM25)
            sparse_results = retriever._keyword_search(question, top_k=top_k)
            # 合并结果并去重
            merged_results = []
            seen_ids = set()
            # 处理向量检索结果
            for doc in vector_results:
                doc_id = doc.metadata.get('id', doc.page_content[:50])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged_results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'source': 'vector'
                    })
            # 处理稀疏检索结果
            for doc in sparse_results:
                doc_id = doc['metadata'].get('id', doc['content'][:50])
                if doc_id not in seen_ids and strategy == "simple_hybrid":
                    # 简易混合模式：添加所有未见过的文档
                    seen_ids.add(doc_id)
                    merged_results.append({
                        'content': doc['content'],
                        'metadata': doc['metadata'],
                        'source': 'sparse'
                    })
                elif strategy == "combined_topk":
                    # 组合top-k模式：即使有重复也添加（稍后去重）
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        merged_results.append({
                            'content': doc['content'],
                            'metadata': doc['metadata'],
                            'source': 'sparse'
                        })
        # 提取检索到的问题
        retrieved_questions = []
        for doc in merged_results:
            # 处理不同类型的结果
            if isinstance(doc, dict):
                # 字典类型结果
                if 'metadata' in doc and isinstance(doc['metadata'], dict):
                    q = doc['metadata'].get('question', '')
                else:
                    q = ''
            elif hasattr(doc, 'metadata'):
                # Document对象
                q = doc.metadata.get('question', '')
            elif isinstance(doc, str):
                # 字符串类型结果 (可能是内容本身)
                q = ''
            else:
                # 其他类型，尝试获取question
                try:
                    q = doc.get('question', '')
                except:
                    q = ''
            retrieved_questions.append(q)
        
        # 计算正确检索数量
        true_positives = sum(1 for q in retrieved_questions if q == question)
        
        # 计算精确度和召回率
        precision = true_positives / len(retrieved_questions) if retrieved_questions else 0
        recall = 1 if true_positives > 0 else 0  # 每个问题只有一个正确答案
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        
        # 计算MRR (Mean Reciprocal Rank)
        try:
            # 找到第一个正确答案的位置
            first_correct_idx = next((i for i, q in enumerate(retrieved_questions) if q == question), None)
            if first_correct_idx is not None:
                mrr = 1.0 / (first_correct_idx + 1)  # +1 因为索引从0开始
            else:
                mrr = 0.0
            all_mrr.append(mrr)
        except Exception as e:
            logger.error(f"Error calculating MRR: {e}")
            all_mrr.append(0.0)
        
        if true_positives > 0:
            correct_retrievals += 1
    
    # 计算评估指标
    metrics = {
        'accuracy': correct_retrievals / total_questions,
        'precision': np.mean(all_precision),
        'recall': np.mean(all_recall),
        'f1_score': np.mean(all_f1),
        'total_questions': total_questions,
        'correct_retrievals': correct_retrievals,
        'method': strategy
    }
    
    # 只有在计算MRR的策略中添加MRR指标
    if all_mrr:
        metrics['mrr'] = np.mean(all_mrr)
    
    return metrics


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

    if use_intersection:
        logger.info(f"Evaluating intersection retrieval with top_k={top_k} for each method...")

    for qa_pair in tqdm(qa_pairs, desc="Evaluating hybrid retrieval"):
        question = qa_pair['question']

        # 获取两种检索结果
        semantic_results = hybrid_retriever._semantic_search(question, top_k * 2)
        keyword_results = hybrid_retriever._keyword_search(question, top_k * 2)
        
        # 根据指定策略选择结果
        if use_intersection:

            # 使用交集策略
            merged_results = hybrid_retriever._find_intersection(semantic_results, keyword_results, top_k)

            # print('len(use_intersection):', len(merged_results))
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
        'correct_retrievals': correct_retrievals,
        'method': 'intersection' if use_intersection else f'hybrid_alpha_{alpha}'
    }
    
    # 添加交集统计信息
    if use_intersection and intersection_stats['total_intersections'] > 0:
        metrics['intersection_stats'] = {
            'total_intersections': intersection_stats['total_intersections'],
            'avg_intersection_size': intersection_stats['total_intersections'] / total_questions,
            'intersection_accuracy': intersection_stats['intersection_correct'] / total_questions
        }

    return metrics


# def evaluate_multihop_retrieval(vector_store, qa_pairs: List[Dict], top_k: int, hops: int = 2, method: str = "frequency", use_intersection: bool = False):
#     """
#     评估多跳检索的性能
#
#     Args:
#         vector_store: FAISS向量存储
#         qa_pairs: 问答对列表
#         top_k: 返回的top k个结果
#         hops: 检索跳数
#         method: 查询增强方法，可选: "frequency", "entity_extraction", "keyword_extraction", "summary"
#         use_intersection: 是否使用交集检索
#     """
#     from retrieval.multetop_retrieval import MultiHopRetriever
#
#     # 初始化多跳检索器
#     try:
#         logger.info("初始化多跳检索器...")
#         multihop_retriever = MultiHopRetriever(
#             vector_store=vector_store,
#             use_reranker=False,
#             hops=hops,
#             use_intersection=use_intersection
#         )
#         logger.info("多跳检索器初始化成功")
#     except Exception as e:
#         logger.error(f"多跳检索器初始化失败: {e}")
#         # 返回空结果
#         return {
#             'accuracy': 0.0,
#             'precision': 0.0,
#             'recall': 0.0,
#             'f1_score': 0.0,
#             'mrr': 0.0
#         }
#
#     total_questions = len(qa_pairs)
#     correct_retrievals = 0
#     all_precision = []
#     all_recall = []
#     all_f1 = []
#     all_mrr = []  # 平均倒数排名
#
#     logger.info(f"Evaluating multi-hop retrieval with hops={hops}, top_k={top_k}, method={method}, use_intersection={use_intersection}...")
#
#     hop_improvements = {
#         'first_hop_correct': 0,      # 第一跳就找到正确答案的数量
#         'second_hop_correct': 0,     # 第二跳找到正确答案的数量(第一跳未找到)
#         'improvement_ratio': 0.0     # 改进比例
#     }
#
#     for i, qa_pair in enumerate(tqdm(qa_pairs, desc="Evaluating multi-hop retrieval")):
#         question = qa_pair['question']
#         logger.info(f"处理问题 {i+1}/{total_questions}: {question[:50]}...")
#
#         # 第一跳检索 (仅用于统计比较)
#         try:
#             logger.info("执行第一跳向量检索...")
#             first_hop_results = vector_store.similarity_search(question, k=top_k)
#             logger.info(f"第一跳检索结果数量: {len(first_hop_results)}")
#
#             first_hop_questions = [
#                 doc.metadata.get('question', '')
#                 for doc in first_hop_results[:top_k]
#             ]
#         except Exception as e:
#             logger.error(f"第一跳向量检索失败: {e}")
#             first_hop_results = []
#             first_hop_questions = []
#
#         # 检查第一跳是否正确
#         first_hop_correct = question in first_hop_questions
#         if first_hop_correct:
#             hop_improvements['first_hop_correct'] += 1
#             logger.info("第一跳检索正确")
#         else:
#             logger.info("第一跳检索未找到正确答案")
#
#         # 执行完整的多跳检索，调用搜索方法时指定查询增强方法
#         try:
#             logger.info("执行多跳检索...")
#             multihop_results = multihop_retriever.search(question, top_k=top_k, method=method)
#             logger.info(f"多跳检索结果数量: {len(multihop_results)}")
#         except Exception as e:
#             logger.error(f"多跳检索失败: {e}")
#             multihop_results = []
#
#         # 提取检索到的问题
#         retrieved_questions = []
#         for doc in multihop_results[:top_k]:
#             # 处理不同类型的结果
#             if isinstance(doc, dict):
#                 # 字典类型结果
#                 if 'metadata' in doc and isinstance(doc['metadata'], dict):
#                     q = doc['metadata'].get('question', '')
#                 else:
#                     q = ''
#             elif hasattr(doc, 'metadata'):
#                 # Document对象
#                 q = doc.metadata.get('question', '')
#             elif isinstance(doc, str):
#                 # 字符串类型结果 (可能是内容本身)
#                 q = ''
#             else:
#                 # 其他类型，尝试获取question
#                 try:
#                     q = doc.get('question', '')
#                 except:
#                     q = ''
#             retrieved_questions.append(q)
#
#         # 计算正确检索数量
#         true_positives = sum(1 for q in retrieved_questions if q == question)
#         logger.info(f"多跳检索正确数量: {true_positives}")
#
#         # 检查多跳后是否正确
#         if true_positives > 0:
#             correct_retrievals += 1
#             logger.info("多跳检索正确")
#             # 如果第一跳没找到但多跳找到了，计为第二跳改进
#             if not first_hop_correct:
#                 hop_improvements['second_hop_correct'] += 1
#                 logger.info("第二跳改进检索结果")
#         else:
#             logger.info("多跳检索未找到正确答案")
#
#         # 计算精确度和召回率
#         precision = true_positives / len(retrieved_questions) if retrieved_questions else 0
#         recall = 1 if true_positives > 0 else 0  # 每个问题只有一个正确答案
#
#         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#
#         all_precision.append(precision)
#         all_recall.append(recall)
#         all_f1.append(f1)
#
#         # 计算MRR (Mean Reciprocal Rank)
#         try:
#             # 找到第一个正确答案的位置
#             first_correct_idx = next((i for i, q in enumerate(retrieved_questions) if q == question), None)
#             if first_correct_idx is not None:
#                 mrr = 1.0 / (first_correct_idx + 1)  # +1 因为索引从0开始
#             else:
#                 mrr = 0.0
#             all_mrr.append(mrr)
#         except Exception as e:
#             logger.error(f"Error calculating MRR: {e}")
#             all_mrr.append(0.0)
#
#     # 计算评估指标
#     metrics = {
#         'accuracy': correct_retrievals / total_questions,
#         'precision': np.mean(all_precision),
#         'recall': np.mean(all_recall),
#         'f1_score': np.mean(all_f1),
#         'mrr': np.mean(all_mrr),
#         'total_questions': total_questions,
#         'correct_retrievals': correct_retrievals,
#         'method': f'multihop_{hops}hops_{method}'
#     }
#
#     # 计算第二跳带来的改进率
#     if hop_improvements['first_hop_correct'] > 0:
#         hop_improvements['improvement_ratio'] = hop_improvements['second_hop_correct'] / (total_questions - hop_improvements['first_hop_correct'])
#
#     # 添加多跳统计信息
#     metrics['hop_improvements'] = hop_improvements
#
#     return metrics


def evaluate_advanced_hybrid_retrieval(vector_store, qa_pairs: List[Dict], top_k: int, 
                                      vector_weight: float = 0.7, 
                                      use_reranker: bool = False,
                                      initial_pool_size: int = 50):
    """
    评估高级混合检索策略的性能
    该策略结合了多种优化技术：
    1. 为向量和稀疏检索结果分配不同权重
    2. 根据检索方法的来源和排名计算综合得分
    3. 使用更大的初始检索池
    4. 考虑文档相似度进行智能去重
    5. 可选的重排序步骤
    
    Args:
        vector_store: 向量存储
        qa_pairs: 问答对列表
        top_k: 最终返回的结果数量
        vector_weight: 向量检索结果的权重 (0-1)
        use_reranker: 是否使用重排序
        initial_pool_size: 初始检索池大小
        
    Returns:
        Dict: 评估指标
    """
    # 初始化混合检索器
    from retrieval.hybrid_retrieval import HybridRetriever
    retriever = HybridRetriever(vector_store, use_reranker=use_reranker)
    
    # 如果启用重排序，确保加载交叉编码器
    if use_reranker:
        try:
            from sentence_transformers import CrossEncoder
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Loaded cross-encoder for reranking")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            use_reranker = False
    
    total_questions = len(qa_pairs)
    correct_retrievals = 0
    all_precision = []
    all_recall = []
    all_f1 = []
    all_mrr = []
    
    logger.info(f"Evaluating advanced hybrid retrieval with top_k={top_k}, vector_weight={vector_weight}...")
    
    for qa_pair in tqdm(qa_pairs, desc="Evaluating advanced hybrid"):
        question = qa_pair['question']
        
        # 1. 使用更大的初始池获取检索结果
        vector_results = vector_store.similarity_search(question, k=initial_pool_size)
        sparse_results = retriever._keyword_search(question, top_k=initial_pool_size)
        # 2. 标准化和处理结果
        processed_results = []
        seen_ids = set()
        # 处理向量检索结果
        for rank, doc in enumerate(vector_results):
            doc_id = doc.metadata.get('id', doc.page_content[:50])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                # 计算基于排名的分数 (排名越高分数越高)
                rank_score = 1.0 / (rank + 1)
                processed_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'vector_score': rank_score,
                    'sparse_score': 0.0,
                    'vector_rank': rank + 1,
                    'sparse_rank': None,
                    'source': 'vector'
                })
        
        # 处理稀疏检索结果
        for rank, doc in enumerate(sparse_results):
            doc_id = doc['metadata'].get('id', doc['content'][:50])
            
            # 检查是否已存在
            existing_idx = None
            for idx, existing in enumerate(processed_results):
                existing_id = existing['metadata'].get('id', existing['content'][:50])
                if existing_id == doc_id:
                    existing_idx = idx
                    break
            
            if existing_idx is not None:
                # 更新已存在的结果
                rank_score = 1.0 / (rank + 1)
                processed_results[existing_idx]['sparse_score'] = rank_score
                processed_results[existing_idx]['sparse_rank'] = rank + 1
                processed_results[existing_idx]['source'] = 'both'
            elif doc_id not in seen_ids:
                # 添加新结果
                seen_ids.add(doc_id)
                rank_score = 1.0 / (rank + 1)
                processed_results.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'vector_score': 0.0,
                    'sparse_score': rank_score,
                    'vector_rank': None,
                    'sparse_rank': rank + 1,
                    'source': 'sparse'
                })
        
        # 3. 计算加权综合得分
        for result in processed_results:
            # 基础得分计算
            weighted_score = (
                vector_weight * result['vector_score'] + 
                (1 - vector_weight) * result['sparse_score']
            )
            
            # 对同时出现在两种结果中的文档给予奖励
            if result['source'] == 'both':
                # 奖励系数基于两个排名的和的倒数
                rank_sum = (result['vector_rank'] or initial_pool_size) + (result['sparse_rank'] or initial_pool_size)
                bonus = 1.0 + (1.0 / rank_sum)  # 排名越高，奖励越大
                weighted_score *= bonus
            
            result['final_score'] = weighted_score
        
        # 4. 根据最终得分排序
        sorted_results = sorted(processed_results, key=lambda x: x['final_score'], reverse=True)
        # 5. 可选的重排序步骤
        if use_reranker:
            try:
                # 准备前top_k*2个结果进行重排序
                candidates = sorted_results[:min(top_k*2, len(sorted_results))]
                pairs = [(question, result['content']) for result in candidates]
                # 使用交叉编码器计算相关性分数
                rerank_scores = cross_encoder.predict(pairs)
                # 更新分数并重新排序
                for i, score in enumerate(rerank_scores):
                    candidates[i]['rerank_score'] = score
                
                reranked_results = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
                final_results = reranked_results[:top_k]
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                final_results = sorted_results[:top_k]
        else:
            final_results = sorted_results[:top_k]
        
        # 6. 提取检索到的问题
        retrieved_questions = []
        for doc in final_results:
            # 处理不同类型的结果
            if isinstance(doc, dict):
                # 字典类型结果
                if 'metadata' in doc and isinstance(doc['metadata'], dict):
                    q = doc['metadata'].get('question', '')
                else:
                    q = ''
            elif hasattr(doc, 'metadata'):
                # Document对象
                q = doc.metadata.get('question', '')
            elif isinstance(doc, str):
                # 字符串类型结果 (可能是内容本身)
                q = ''
            else:
                # 其他类型，尝试获取question
                try:
                    q = doc.get('question', '')
                except:
                    q = ''
            retrieved_questions.append(q)
        
        # 7. 计算评估指标
        true_positives = sum(1 for q in retrieved_questions if q == question)
        
        precision = true_positives / len(retrieved_questions) if retrieved_questions else 0
        recall = 1 if true_positives > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        
        # 计算MRR
        try:
            first_correct_idx = next((i for i, q in enumerate(retrieved_questions) if q == question), None)
            if first_correct_idx is not None:
                mrr = 1.0 / (first_correct_idx + 1)
            else:
                mrr = 0.0
            all_mrr.append(mrr)
        except Exception as e:
            logger.error(f"Error calculating MRR: {e}")
            all_mrr.append(0.0)
        
        if true_positives > 0:
            correct_retrievals += 1
    
    # 计算评估指标
    metrics = {
        'accuracy': correct_retrievals / total_questions,
        'precision': np.mean(all_precision),
        'recall': np.mean(all_recall),
        'f1_score': np.mean(all_f1),
        'mrr': np.mean(all_mrr),
        'total_questions': total_questions,
        'correct_retrievals': correct_retrievals,
        'method': 'advanced_hybrid'
    }

    return metrics


def evaluate_llm_reranker_retrieval(vector_store, qa_pairs: List[Dict], top_k: int, initial_pool_size: int = 5,
                                    model_name: str = "Qwen2.5-7B-Instruct-local", api_base_url: str = "http://localhost:8000/v1",
                                    scoring_mode: str = "relative"):
    """
    评估使用LLM重排序器的检索性能
    
    Args:
        vector_store: 向量存储
        qa_pairs: 问答对列表
        top_k: 最终返回的结果数量
        initial_pool_size: 初始检索池大小，即传递给LLM重排序的候选文档数量
        model_name: LLM模型名称
        api_base_url: API基础URL
        scoring_mode: 评分模式，'relative'让LLM选择最佳文档，'absolute'让LLM给每个文档打分
        
    Returns:
        Dict: 评估指标
    """
    total_questions = len(qa_pairs)
    correct_retrievals = 0
    all_precision = []
    all_recall = []
    all_f1 = []
    all_mrr = []
    
    logger.info(f"Evaluating LLM reranker retrieval with scoring_mode={scoring_mode}, initial_pool_size={initial_pool_size}, top_k={top_k}...")
    
    # 初始化LLM重排序器
    reranker = LLMReranker(
        model_name=model_name,
        api_base_url=api_base_url,
        temperature=0.1,
        scoring_mode=scoring_mode,
        max_tokens=400  # 增加token数，特别是对于absolute模式需要更多输出空间
    )
    
    for idx, qa_pair in tqdm(enumerate(qa_pairs), total=total_questions, desc="Evaluating LLM reranker"):
        question = qa_pair['question']
        
        # 获取向量检索结果作为初始候选池
        results = vector_store.similarity_search(question, k=initial_pool_size)
        
        # 准备重排序的文档格式
        documents_for_rerank = []
        for doc in results:
            documents_for_rerank.append({
                "page_content": doc.page_content, # 用于匹配的文本
                "metadata": doc.metadata
            })
        
        # 使用LLM重排序
        reranked_documents = reranker.rerank(question, documents_for_rerank, top_k=top_k)
        
        # 提取检索到的问题
        retrieved_questions = []
        for doc in reranked_documents:
            retrieved_questions.append(doc.metadata.get('question', ''))
        
        # 计算正确检索数量
        true_positives = sum(1 for q in retrieved_questions if q == question)
        
        # 计算精确度和召回率
        precision = true_positives / len(retrieved_questions) if retrieved_questions else 0
        recall = 1 if true_positives > 0 else 0
        
        # 计算F1分数
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
            
        # 计算MRR
        try:
            first_correct_idx = next((i for i, q in enumerate(retrieved_questions) if q == question), None)
            if first_correct_idx is not None:
                mrr = 1.0 / (first_correct_idx + 1)
            else:
                mrr = 0.0
            all_mrr.append(mrr)
        except Exception as e:
            logger.error(f"Error calculating MRR: {e}")
            all_mrr.append(0.0)
        
        # 更新统计结果
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        
        if true_positives > 0:
            correct_retrievals += 1
    
    # 计算平均指标
    metrics = {
        'accuracy': correct_retrievals / total_questions,
        'precision': np.mean(all_precision),
        'recall': np.mean(all_recall),
        'f1_score': np.mean(all_f1),
        'mrr': np.mean(all_mrr),
        'method': f'llm_reranker_{scoring_mode}',  # 在方法名中包含评分模式
        'total_questions': total_questions,
        'correct_retrievals': correct_retrievals
    }
    
    logger.info(f"LLM Reranker ({scoring_mode}) Retrieval Results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"MRR: {metrics['mrr']:.4f}")
    
    return metrics


def compare_all_retrieval_methods(vector_store, qa_pairs: List[Dict], top_k: int):
    """
    比较所有检索方法的性能（原始、混合、高级混合、交集、多跳）
    """
    results = {}
    # 1. 评估向量检索方法
    results['vector'] = evaluate_retrieval(vector_store, qa_pairs, top_k)
    # 2. 评估稀疏检索方法(BM25)
    results['sparse'] = evaluate_retrieval_strategy(vector_store, qa_pairs, top_k, strategy="sparse")
    # 3. 评估简易混合检索方法
    # results['simple_hybrid'] = evaluate_retrieval_strategy(vector_store, qa_pairs, top_k, strategy="simple_hybrid")
    # 4. 评估组合Top-K检索方法
    # results['combined_topk'] = evaluate_retrieval_strategy(vector_store, qa_pairs, top_k, strategy="combined_topk")

    # 5. 评估高级混合检索方法（不同权重配置）
    # vector_weights = [0.6, 0.7, 0.8]
    # pool_sizes = [50]
    # for weight in vector_weights:
    #     for pool_size in pool_sizes:
    #         method_name = f'adv_hybrid_w{weight}_p{pool_size}'
    #         logger.info(f"Evaluating advanced hybrid retrieval with weight={weight}, pool_size={pool_size}...")
    #         results[method_name] = evaluate_advanced_hybrid_retrieval(vector_store, qa_pairs, top_k, vector_weight=weight, initial_pool_size=pool_size)
    # 6. 评估交集检索方法
    # results['intersection'] = evaluate_hybrid_retrieval( vector_store, qa_pairs, top_k, alpha=0.7, use_intersection=True)
    # 7. 评估LLM重排序检索方法
    # try:
    #     logger.info("Evaluating LLM reranker retrieval method...")
    #     results['llm_reranker'] = evaluate_llm_reranker_retrieval(
    #         vector_store,
    #         qa_pairs,
    #         top_k=top_k,
    #         initial_pool_size=10,  # 初始候选池大小
    #         model_name="Qwen2.5-7B-Instruct-local",
    #         api_base_url="http://localhost:8000/v1",
    #         scoring_mode="relative"  # 添加scoring_mode参数
    #     )
    # except Exception as e:
    #     logger.error(f"Failed to evaluate LLM reranker: {e}")
    try:
        logger.info("Evaluating LLM reranker retrieval method...")
        results['llm_reranker'] = evaluate_llm_reranker_retrieval(
            vector_store,
            qa_pairs,
            top_k=top_k,
            initial_pool_size=10,  # 初始候选池大小
            model_name="Qwen2.5-7B-Instruct-reranker",
            api_base_url="http://localhost:8000/v1",
            scoring_mode="absolute"  # 添加scoring_mode参数
        )
    except Exception as e:
        logger.error(f"Failed to evaluate LLM reranker: {e}")

    # # 8. 评估多跳检索（使用交集检索）
    # logger.info("Evaluating multi-hop retrieval method...")
    # results['multihop_intersection'] = evaluate_multihop_retrieval(
    #     vector_store, qa_pairs, top_k, hops=2, method="frequency", use_intersection=True
    # )

    return results


def plot_comparison_with_hyde(results: Dict):
    """
    可视化所有检索方法的性能比较，包括HyDE
    """
    try:
        import matplotlib.pyplot as plt

        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        methods = list(results.keys())

        # 检查metrics是否存在于结果中，如果没有则跳过
        valid_metrics = []
        for metric in metrics_names:
            if all(metric in results[method] for method in methods):
                valid_metrics.append(metric)

        # 更新指标列表
        metrics_names = valid_metrics

        # 创建图形
        fig = plt.figure(figsize=(15, 10))

        # 1. 柱状图
        ax1 = plt.subplot(211)
        x = np.arange(len(methods))
        width = 0.15

        # 设置颜色映射
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        for i, metric in enumerate(metrics_names):
            values = [results[method].get(metric, 0) for method in methods]
            ax1.bar(x + i * width, values, width, label=metric, color=colors[i % len(colors)])

        ax1.set_ylabel('Score')
        ax1.set_title('Comparison of All Retrieval Methods')
        ax1.set_xticks(x + width * (len(metrics_names) - 1) / 2)
        ax1.set_xticklabels([method.replace('_', ' ').title() for method in methods], rotation=45)
        ax1.legend()

        # 显示具体数值
        for i, metric in enumerate(metrics_names):
            for j, method in enumerate(methods):
                value = results[method].get(metric, 0)
                ax1.text(j + i * width, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        # 2. 雷达图
        ax2 = plt.subplot(212, projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        # 添加图例填充和标签
        for i, method in enumerate(methods):
            values = [results[method].get(metric, 0) for metric in metrics_names]
            values = np.concatenate((values, [values[0]]))
            color = colors[i % len(colors)]
            ax2.plot(angles, values, 'o-', label=method.replace('_', ' ').title(), color=color)
            ax2.fill(angles, values, alpha=0.25, color=color)

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics_names])
        ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # 添加标题
        plt.suptitle('Retrieval Methods Performance Comparison', fontsize=16)

        plt.tight_layout()
        plt.savefig('retrieval_comparison.png', dpi=300)
        plt.close()

        logger.info("Comparison plot saved as 'retrieval_comparison.png'")

        # 额外创建一个专注于交集搜索结果的图表（如果存在）
        if 'intersection' in results and 'intersection_stats' in results['intersection']:
            try:
                stats = results['intersection']['intersection_stats']

                # 创建新图形
                fig, ax = plt.subplots(figsize=(8, 6))

                # 准备数据
                labels = ['交集准确率', '平均交集大小/5', '总交集数量/总问题数']
                values = [
                    stats['intersection_accuracy'],
                    min(stats['avg_intersection_size'] / 5, 1.0),  # 除以5是为了缩放到0-1范围
                    min(stats['total_intersections'] / len(results['intersection'].get('total_questions', 100)), 1.0)
                ]

                # 绘制条形图
                ax.bar(labels, values, color=['#3498db', '#2ecc71', '#f39c12'])

                # 添加数值标签
                for i, v in enumerate(values):
                    ax.text(i, v + 0.02, f'{v:.3f}', ha='center')

                # 添加标题和标签
                ax.set_title('交集搜索统计信息')
                ax.set_ylim(0, 1.1)
                ax.set_ylabel('数值')

                # 保存图表
                plt.tight_layout()
                plt.savefig('intersection_stats.png', dpi=300)
                plt.close()

                logger.info("Intersection statistics plot saved as 'intersection_stats.png'")
            except Exception as e:
                logger.error(f"Error creating intersection stats plot: {e}")

    except ImportError:
        logger.warning("matplotlib not installed. Skipping visualization.")

    # 额外创建一个专注于多跳检索结果的图表（如果存在）
    if 'multihop' in results and 'hop_improvements' in results['multihop']:
        try:
            hop_stats = results['multihop']['hop_improvements']
            total_questions = results['multihop']['total_questions']

            # 创建新图形
            fig, ax = plt.subplots(figsize=(8, 6))

            # 准备数据
            labels = ['首跳正确率', '二跳改进率', '总体正确率']
            values = [
                hop_stats['first_hop_correct'] / total_questions,
                hop_stats['improvement_ratio'],
                results['multihop']['accuracy']
            ]

            # 绘制条形图
            ax.bar(labels, values, color=['#3498db', '#2ecc71', '#f39c12'])

            # 添加数值标签
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center')

            # 添加标题和标签
            ax.set_title('多跳检索性能统计')
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('正确率')

            # 保存图表
            plt.tight_layout()
            plt.savefig('multihop_stats.png', dpi=300)
            plt.close()

            logger.info("Multi-hop statistics plot saved as 'multihop_stats.png'")
        except Exception as e:
            logger.error(f"Error creating multi-hop stats plot: {e}")


def main():
    """
    主函数
    """
    # 加载向量存储
    vector_store = load_vector_store(
        model_path=model_path,
        faiss_index_path=faiss_index_path,
        use_custom_model=True
    )
    logger.info(f"Loaded vector store from {faiss_index_path}")

    # 加载测试数据集
    qa_dataset = load_qa_dataset(test_data_path)


    print(f"Loaded {len(qa_dataset)} QA pairs.")

    # 比较所有检索方法的性能
    all_results = compare_all_retrieval_methods(
        vector_store,
        qa_dataset,
        top_k=top_k,
    )

    # 打印详细结果
    for method, metrics in all_results.items():
        print(f"\n{method.upper()} Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        if 'mrr' in metrics:
            print(f"MRR: {metrics['mrr']:.4f}")

    # 绘制比较图
    # plot_comparison_with_hyde(all_results)


if __name__ == "__main__":
    main()


'''
基于您的项目现有代码和结构，以下几种检索方法与您的项目最为契合：

查询改写和扩展（Query Rewriting & Expansion）：
您已经在_enhance_query方法中实现了一些查询增强功能，可以进一步扩展为更强大的查询改写系统
可以利用您现有的多跳架构，在每一跳之前进行查询改写
RAG增强检索：
与您的多跳检索框架高度兼容，可以在每一跳之间使用生成模型进行中间结果的总结和查询增强
可以在现有的_enhance_query方法基础上实现
自适应检索（Adaptive Retrieval）：
您已经实现了向量检索和交集检索，可以添加一个自适应层来根据查询类型自动选择最佳方法
可以利用您现有的评估框架来训练一个简单的分类器，决定何时使用哪种检索方法
长文本检索优化：
如果您的数据集包含长文档，这将是一个很好的补充
可以在现有的检索框架中添加文档分段和重组逻辑
混合密度检索（Hybrid Dense Retrieval）：
您已经实现了基本的混合检索，可以扩展为使用多种不同的向量模型
可以在现有的HybridRetriever类基础上进行扩展
具体建议实施的优先顺序：

查询改写和扩展：这是最容易集成且可能带来最大收益的方法。您可以使用LLM（如GPT或本地部署的模型）来改写查询，或者实现基于规则的同义词扩展。
RAG增强检索：这可以显著提高多跳检索的质量，特别是在处理复杂查询时。
自适应检索：这可以提高系统的整体效率和准确性，让系统根据查询特点自动选择最佳的检索策略。
这些方法都可以在您现有的代码基础上实现，不需要大规模重构，同时能够显著提高检索性能。
'''