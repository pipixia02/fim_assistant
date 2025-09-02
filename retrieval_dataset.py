import json
import random
from typing import List, Dict, Tuple
import os
from sklearn.model_selection import train_test_split
import logging
import nltk
from rank_bm25 import BM25Okapi
import numpy as np
from loguru import logger

nltk.data.path.append('/home/hongchang/llm_projects/shenhang/nltk_data')

# 下载NLTK所需数据
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')


def load_qa_dataset(file_path: str) -> List[Dict]:
    """
    加载QA数据集
    Args:
        file_path: QA数据集的文件路径
    Returns:
        包含QA对的列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} QA pairs from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        raise


def create_retrieval_dataset(qa_data: List[Dict], dataset, seed: int = 42) -> List[Dict]:
    """
    创建检索数据集
    Args:
        qa_data: QA数据集
        dataset: 要处理的数据集子集
        seed: 随机种子，用于保证结果的可重复性
    Returns:
        检索数据集列表
    """
    # 设置随机种子以确保可重复性
    random.seed(seed)
    
    retrieval_data = []
    all_descriptions = [item['task']+' '+item['original_description'] for item in qa_data]
    #  all_descriptions = [item['task']+' '+item['summarized_description'] for item in qa_data]
    for item in dataset:
        # 创建一个不包含当前description的列表
        other_descriptions = [desc for desc in all_descriptions if desc != item['task'] + ' ' + item['original_description']]
        # other_descriptions = [desc for desc in all_descriptions  if desc != item['task']+' '+item['summarized_description']]
        
        # 随机选择一个不相关的文档
        irrelevant_doc = random.choice(other_descriptions)
        
        # 创建检索数据项
        retrieval_item = {
            'question': item['question'],
            'relevant_doc': item['task'] + ' ' + item['original_description'],
            # 'relevant_doc': item['task']+' '+item['summarized_description'],
            'irrelevant_doc': irrelevant_doc
        }
        retrieval_data.append(retrieval_item)
    
    print(f"Created {len(retrieval_data)} retrieval dataset items")
    return retrieval_data


def split_dataset(data: List[Dict], 
                 train_ratio: float = 0.7,
                 dev_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    将数据集分割为训练集、验证集和测试集
    Args:
        data: 完整数据集
        train_ratio: 训练集比例
        dev_ratio: 验证集比例
    Returns:
        (训练集, 验证集, 测试集)的元组
    """
    # 首先分割出训练集
    train_data, temp_data = train_test_split(
        data, 
        train_size=train_ratio, 
        random_state=42
    )
    
    # 从剩余数据中分割出验证集和测试集
    dev_ratio_adjusted = dev_ratio / (1 - train_ratio)
    dev_data, test_data = train_test_split(
        temp_data, 
        train_size=dev_ratio_adjusted, 
        random_state=42
    )
    
    print(f"Split dataset into {len(train_data)} train, "f"{len(dev_data)} dev, {len(test_data)} test samples")
    return train_data, dev_data, test_data


def save_dataset(data: List[Dict], file_path: str):
    """
    保存数据集到文件
    Args:
        data: 要保存的数据
        file_path: 保存路径
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved {len(data)} items to {file_path}")
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")
        raise


def create_hard_samples_with_bm25(qa_data: List[Dict], dataset: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    创建检索数据集，使用BM25算法检索相似的但错误的文档作为难样本
    
    Args:
        qa_data: 所有QA数据集
        dataset: 需要创建难样本的数据集子集
        top_k: 选择前k个最相似文档
    
    Returns:
        包含难样本的检索数据集
    """
    # 准备所有文档和其id映射
    all_docs = []
    doc_to_id = {}
    
    for i, item in enumerate(qa_data):
        doc = item['task'] + ' ' + item['original_description']
        all_docs.append(doc)
        doc_to_id[doc] = i
    
    # 初始化BM25
    try:
        # 分词
        tokenized_docs = []
        for doc in all_docs:
            tokens = nltk.word_tokenize(doc.lower())
            tokenized_docs.append(tokens)
        
        # 创建BM25检索器
        bm25 = BM25Okapi(tokenized_docs)
        print(f"Successfully initialized BM25 with {len(tokenized_docs)} documents")
    except Exception as e:
        print(f"Error initializing BM25: {e}")
        raise
    
    retrieval_data = []
    
    for item in dataset:
        # 获取当前问题和正确文档
        question = item['question']
        current_doc = item['task'] + ' ' + item['original_description']
        current_doc_id = doc_to_id[current_doc]
        
        # 对问题进行分词
        tokenized_query = nltk.word_tokenize(question.lower())
        
        # 使用BM25获取相似文档的分数
        scores = bm25.get_scores(tokenized_query)
        
        # 为了避免选择问题自身的文档，将其分数设为-1
        scores[current_doc_id] = -1
        
        # 获取top_k个最相似的文档索引
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # 从top_k个最相似文档中随机选择一个作为难样本
        hard_sample_id = random.choice(top_indices)

        hard_sample_doc = all_docs[hard_sample_id]
        for index in top_indices:
            if index == current_doc_id:
                continue
            hard_sample_doc = all_docs[index]
            retrieval_item = {
                'question': question,
                'relevant_doc': current_doc,
                'irrelevant_doc': hard_sample_doc,
                'similarity_score': float(scores[hard_sample_id])  # 记录相似度得分用于分析
            }
            retrieval_data.append(retrieval_item)
            # 创建检索数据项

    
    logger.info(f"Created {len(retrieval_data)} hard sample retrieval dataset items using BM25")
    return retrieval_data


def convert_question_format(file_path: str):
    """
    将 hyde_gen.json 中的 question 字段从列表转换为字符串

    Args:
        file_path: JSON文件路径
    """
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 转换格式
    for item in data:
        if isinstance(item['question'], list) and len(item['question']) > 0:
            item['question'] = item['question'][0]

    # 保存修改后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Successfully converted question format in {file_path}")


def deduplicate_questions(input_file: str, output_file: str):
    """
    从输入文件中读取数据，确保每个不同的"question"只有一个条目，
    并将结果保存到输出文件中

    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
    """
    print(f"Processing {input_file}...")

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {input_file}")

    # 使用字典存储每个问题的第一个出现
    unique_questions = {}

    # 遍历数据，保留每个问题的第一个实例
    for item in data:
        question = item['question']
        if question not in unique_questions:
            unique_questions[question] = item

    # 转换回列表
    result = list(unique_questions.values())

    print(f"Reduced to {len(result)} unique questions")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {output_file}")

    return len(data), len(result)


def main():
    """
    主函数
    """
    input_file = 'dataset/retrieval_data/new2/train.json'
    output_file = 'dataset/retrieval_data/new2/single_train.json'

    total_count, unique_count = deduplicate_questions(input_file, output_file)

    print(f"Summary: Reduced {total_count} entries to {unique_count} unique entries")
    print(f"Removed {total_count - unique_count} duplicate questions")


def main_base(qa_data, train_qa_data, dev_qa_data, test_qa_data, output_dir):
    # 设置全局随机种子以确保结果可重复
    global_seed = 42
    random.seed(global_seed)
    
    all_train_ret_data, all_dev_data = [], []

    dev_ret_data = create_retrieval_dataset(qa_data, dev_qa_data, seed=global_seed)
    
    train_ret_data = create_retrieval_dataset(qa_data, train_qa_data, seed=global_seed)
    
    for i in range(100):
        # 为每次迭代使用不同但确定的种子
        iter_seed = global_seed + i
        train_ret_data = create_retrieval_dataset(qa_data, train_qa_data, seed=iter_seed)
        all_train_ret_data.extend(train_ret_data)

    test_ret_data = create_retrieval_dataset(qa_data, test_qa_data, seed=global_seed)

    save_dataset(train_qa_data, os.path.join(output_dir, 'qa_train.json'))
    save_dataset(dev_qa_data, os.path.join(output_dir, 'qa_dev.json'))
    save_dataset(test_qa_data, os.path.join(output_dir, 'qa_test.json'))

    save_dataset(train_ret_data, os.path.join(output_dir, 'single_train.json'))
    save_dataset(all_train_ret_data, os.path.join(output_dir, 'train.json'))
    save_dataset(dev_ret_data, os.path.join(output_dir, 'dev.json'))
    save_dataset(test_ret_data, os.path.join(output_dir, 'test.json'))

    print("\nDataset statistics:")
    print(f"Total samples: {len(qa_data)}")
    print(f"Train samples: {len(train_qa_data)} ({len(train_qa_data) / len(qa_data) * 100:.1f}%)， Total train: {len(all_train_ret_data)}")
    print(f"Dev samples: {len(dev_qa_data)} ({len(dev_qa_data) / len(qa_data) * 100:.1f}%)")
    print(f"Test samples: {len(test_qa_data)} ({len(test_qa_data) / len(qa_data) * 100:.1f}%)")


def main_hard(qa_data, train_data, dev_data, test_data, output_dir):


    top_k = 10

    # 使用BM25创建难样本
    all_train_data = create_hard_samples_with_bm25(qa_data, train_data, top_k=top_k)
    all_dev_data = create_hard_samples_with_bm25(qa_data, dev_data, top_k=top_k)
    all_test_data = create_hard_samples_with_bm25(qa_data, test_data, top_k=top_k)
    
    # 保存数据集
    save_dataset(all_train_data, os.path.join(output_dir, 'bm25_train.json'))
    save_dataset(all_dev_data, os.path.join(output_dir, 'bm25_dev.json'))
    save_dataset(all_test_data, os.path.join(output_dir, 'bm25_test.json'))
    
    logger.info("\nDataset statistics:")
    logger.info(f"Total samples: {len(qa_data)}")
    logger.info(f"Train samples: {len(train_data)} ({len(train_data)/len(qa_data)*100:.1f}%)")
    logger.info(f"Dev samples: {len(dev_data)} ({len(dev_data)/len(qa_data)*100:.1f}%)")
    logger.info(f"Test samples: {len(test_data)} ({len(test_data)/len(qa_data)*100:.1f}%)")
    
    logger.info(f"Total train hard samples: {len(all_train_data)}")
    logger.info(f"Total dev hard samples: {len(all_dev_data)}")
    logger.info(f"Total test hard samples: {len(all_test_data)}")


if __name__ == '__main__':

    # 设置输入输出路径
    input_file = 'dataset/qa_data/qa_prompt1_with_summarise.json'  # 使用新的数据集
    output_dir = 'dataset/retrieval_data/prompt1/summarise'
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 加载QA数据集
    qa_data = load_qa_dataset(input_file)
    # train_data, dev_data, test_data = split_dataset(
    #     qa_data,
    #     train_ratio=0.7,
    #     dev_ratio=0.15
    # )
    train_data = load_qa_dataset('dataset/retrieval_data/prompt1/summarise/qa_train.json')
    dev_data = load_qa_dataset('dataset/retrieval_data/prompt1/summarise/qa_dev.json')
    test_data = load_qa_dataset('dataset/retrieval_data/prompt1/summarise/qa_test.json')

    # save_dataset(train_data, os.path.join(output_dir, 'train.json'))

    main_base(qa_data, train_data, dev_data, test_data, output_dir)
    # main_hard(qa_data, train_data, dev_data, test_data, output_dir)
