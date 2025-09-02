"""
简单测试LLM重排器的效果，评估模型是否能找出最相关文档
"""

import json
import time
import random
from typing import List, Dict
from tqdm import tqdm
from loguru import logger
from retrieval.llm_reranker import LLMReranker

# 配置参数
LORA_TRAIN_DATA_PATH = "dataset/llm_reranker/lora_reranker.json"  # 训练数据路径
TEST_SAMPLE_SIZE = 100  # 测试样本数量

def load_test_data(file_path: str, sample_size: int) -> List[Dict]:
    """从训练数据中加载测试样本"""
    try:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保sample_size不超过数据集大小
        sample_size = min(sample_size, len(data))
        
        # 随机采样
        test_samples = random.sample(data, sample_size)
        logger.info(f"Successfully loaded {len(test_samples)} test samples")
        return test_samples
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return []

def parse_data(sample: Dict) -> tuple:
    """解析样本数据，提取问题、文档和正确答案"""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output_text = sample.get("output", "")
    
    # 从input中提取问题和文档
    lines = input_text.split('\n')
    question = ""
    documents = []
    current_doc = ""
    doc_id = None
    
    for line in lines:
        if line.startswith("Question:"):
            question = line[len("Question:"):].strip()
        elif line.startswith("Document "):
            # 保存上一个文档
            if doc_id is not None and current_doc:
                documents.append({"id": doc_id, "content": current_doc.strip()})
            
            # 开始新文档
            parts = line.split(":")
            doc_id = int(parts[0].replace("Document ", "").strip())
            current_doc = parts[1].strip() if len(parts) > 1 else ""
        elif doc_id is not None:
            # 继续累积当前文档的内容
            current_doc += "\n" + line
    
    # 添加最后一个文档
    if doc_id is not None and current_doc:
        documents.append({"id": doc_id, "content": current_doc.strip()})
    
    # 从output中提取最佳文档ID
    best_doc_id = None
    highest_score = -1
    
    if output_text:
        output_lines = output_text.split('\n')
        for line in output_lines:
            if ":" in line:
                parts = line.split(":")
                doc_num = int(parts[0].replace("Document ", "").strip())
                score = float(parts[1].strip())
                if score > highest_score:
                    highest_score = score
                    best_doc_id = doc_num
    
    return question, documents, best_doc_id

def test_reranker(test_samples: List[Dict]):
    """测试重排器模型"""
    logger.info("Initializing LLM Reranker")
    
    # 初始化重排器
    reranker = LLMReranker(
        model_name="Qwen2.5-7B-Instruct-reranker",
        api_base_url="http://localhost:8000/v1",
        temperature=0.1,
        max_tokens=1024,
        max_retries=2,
        scoring_mode="absolute"  # 使用绝对评分模式
    )
    
    correct_count = 0
    total_count = 0
    results = []
    
    for i, sample in enumerate(tqdm(test_samples, desc="Testing reranker")):
        try:
            # 解析样本
            question, documents, best_doc_id = parse_data(sample)
            
            if not question or not documents or best_doc_id is None:
                logger.warning(f"Skipping sample {i}: Invalid format")
                continue
            
            # 准备重排器输入
            reranker_input = []
            for doc in documents:
                reranker_input.append({
                    "page_content": doc["content"],
                    "metadata": {"id": doc["id"]}
                })
            
            # 使用重排器进行排序
            try:
                start_time = time.time()
                ranked_docs = reranker.rerank(question, reranker_input, top_k=len(reranker_input))
                print('ranked_docs:', ranked_docs)
                print('*'*90)
                elapsed_time = time.time() - start_time
                
                # 获取排名结果
                ranked_ids = [doc.metadata.get("id") for doc in ranked_docs]
                ranked_scores = [doc.score for doc in ranked_docs]
                
                # 判断最高得分的文档是否正确
                top_doc_id = ranked_ids[0]
                is_correct = (top_doc_id == best_doc_id)
                
                if is_correct:
                    correct_count += 1
                
                total_count += 1
                
                # 保存结果
                results.append({
                    "question": question,
                    "best_doc_id": best_doc_id,
                    "model_prediction": top_doc_id,
                    "is_correct": is_correct,
                    "all_scores": list(zip(ranked_ids, ranked_scores)),
                    "time": elapsed_time
                })
                
                # 实时打印每个样本的结果
                status = "✓" if is_correct else "✗"
                print(f"Sample {i+1}: {status} Model predicted: Doc {top_doc_id}, Ground truth: Doc {best_doc_id}")
                
            except Exception as e:
                logger.error(f"Error during reranking for sample {i}: {e}")
                continue
                
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            continue
    
    # 打印总体结果
    accuracy = correct_count / total_count if total_count > 0 else 0
    print("\n" + "="*50)
    print(f"总体准确率: {accuracy:.2%} ({correct_count}/{total_count})")
    print("="*50)
    
    # 显示一些错误的例子
    # incorrect_examples = [r for r in results if not r["is_correct"]]
    # if incorrect_examples:
    #     print("\n错误预测的例子:")
    #     for i, example in enumerate(incorrect_examples[:3]):  # 显示最多3个错误例子
    #         print(f"\n错误例子 {i+1}:")
    #         print(f"问题: {example['question']}")
    #         print(f"正确文档ID: {example['best_doc_id']}")
    #         print(f"模型预测ID: {example['model_prediction']}")
    #         print("所有文档评分:")
    #         for doc_id, score in example["all_scores"]:
    #             print(f"  文档 {doc_id}: {score}" + (" (正确答案)" if doc_id == example["best_doc_id"] else ""))
    #
    # # 保存结果
    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    # result_file = f"results/simple_reranker_test_{timestamp}.json"
    # with open(result_file, 'w', encoding='utf-8') as f:
    #     json.dump({
    #         "accuracy": accuracy,
    #         "correct_count": correct_count,
    #         "total_count": total_count,
    #         "sample_results": results
    #     }, f, ensure_ascii=False, indent=2)
    #
    # logger.info(f"Results saved to {result_file}")

def main():
    """主函数"""
    # 设置日志
    logger.add("logs/test_reranker_{time}.log")
    logger.info("Starting Simple LLM Reranker Test")
    
    # 确保results目录存在
    import os
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 加载测试数据
    test_samples = load_test_data(LORA_TRAIN_DATA_PATH, TEST_SAMPLE_SIZE)
    if not test_samples:
        logger.error("No test samples loaded. Exiting.")
        return
    
    # 测试重排器
    test_reranker(test_samples)
    
    logger.info("LLM Reranker Test completed")

if __name__ == "__main__":
    main()
