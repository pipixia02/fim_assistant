import copy
import json
import os
import random
from typing import List, Dict
import argparse

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

def create_qwen_chat_format(qa_data: List[Dict], include_procedure: bool = True) -> List[Dict]:
    """
    将QA数据转换为Qwen模型的聊天格式
    
    Args:
        qa_data: 包含问题、程序和答案的QA数据
        include_procedure: 是否在用户输入中包含原始程序/提示信息
        
    Returns:
        适用于Qwen模型微调的聊天格式数据
    """
    formatted_data = []
    
    for item in qa_data:
        # 检查数据项中是否包含所需字段
        if not all(key in item for key in ["question", "answer"]):
            continue
            
        # 准备系统消息
        system_message = "你是一个专业的航空维修顾问，回答问题时要准确、专业且符合航空领域标准。"
        
        # 准备用户消息 (根据include_procedure决定是否包含提示信息)
        if include_procedure and "original_procedure" in item:
            user_message = f"{item['question']}\n\nreference：\n{item['original_procedure']}"
        else:
            user_message = item['question']
            
        # 准备助手回复
        assistant_message = item['answer']
        
        # 创建聊天格式的数据
        chat_item = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
        }
        
        formatted_data.append(chat_item)
    
    return formatted_data

def create_cross_dataset(formatted_data: List[Dict], cross_ratio: float = 0.3) -> List[Dict]:
    """
    创建不包含程序信息的交叉数据集（强制模型使用自身知识）
    
    Args:
        formatted_data: 已格式化的数据
        cross_ratio: 需要移除程序信息的数据比例
        
    Returns:
        混合了有程序和无程序的数据集
    """
    # 深拷贝原始数据
    import copy
    mixed_data = copy.deepcopy(formatted_data)
    
    # 计算需要移除程序信息的样本数
    samples_to_modify = int(len(mixed_data) * cross_ratio)
    indices_to_modify = random.sample(range(len(mixed_data)), samples_to_modify)
    
    # 修改选定的样本，移除参考信息
    for idx in indices_to_modify:
        user_message = mixed_data[idx]["messages"][1]["content"]
        # 如果包含参考信息，则移除
        if "参考信息：" in user_message:
            question_only = user_message.split("\n\n参考信息：")[0]
            mixed_data[idx]["messages"][1]["content"] = question_only
    
    return mixed_data

def split_dataset(data: List[Dict], train_ratio: float = 0.9) -> tuple:
    """
    将数据集分割为训练集和验证集
    
    Args:
        data: 完整数据集
        train_ratio: 训练集比例
        
    Returns:
        (训练集, 验证集)
    """
    # 随机打乱数据
    random.shuffle(data)
    
    # 计算分割点
    split_idx = int(len(data) * train_ratio)
    
    # 分割数据集
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data

def convert_qa_to_instruction_format(qa_data: List[Dict]) -> List[Dict]:
    """
    将qa_train.json中的数据转换为instruction-input-output格式
    
    Args:
        qa_data: 包含问题、初始评估和程序的QA数据
        
    Returns:
        转换后的instruction格式数据
    """
    instruction_data = []
    
    for item in qa_data:
        # 检查数据项中是否包含所需字段
        if not ("original_procedure" in item and "answer" in item):
            continue
        # 创建instruction格式的数据
        if "original_initial_evaluation" in item :
            instruction_item = {
                "instruction": f"Initial Evaluation:{item.get('original_initial_evaluation', '')}\n\nFault Isolation Procedure:{item.get('original_procedure', '')}",
                "input": item["question"],
                "output": item["answer"]
            }
        else:
            instruction_item = {
                "instruction": f"Fault Isolation Procedure:{item.get('original_procedure', '')}",
                "input": item["question"],
                "output": item["answer"]
            }

        instruction_data.append(instruction_item)
    
    return instruction_data

def convert_reranker_data(file_path: str, output_path: str, max_doc_length: int = 512, scoring_mode: str = "absolute"):
    """
    将重排序数据转换为LoRA微调所需的指令格式
    
    Args:
        file_path: 输入重排序数据文件路径
        output_path: 输出指令格式数据文件路径
        max_doc_length: 最大文档长度
        scoring_mode: 评分模式，'absolute'或'relative'
    """
    try:
        # 加载重排序数据
        with open(file_path, 'r', encoding='utf-8') as f:
            reranker_data = json.load(f)
        
        print(f"Successfully loaded {len(reranker_data)} records from {file_path}")


        # reranker_data = augment_reranker_data(reranker_data)

        # 检查分数范围（仅用于调试）
        if scoring_mode == "absolute" and len(reranker_data) > 0:
            # 采样前10条记录查看分数分布
            sample_count = min(10, len(reranker_data))
            print(f"检查前{sample_count}条记录的分数分布：")
            for i in range(sample_count):
                if "scores" in reranker_data[i]:
                    scores = reranker_data[i]["scores"]
                    print(f"记录 {i+1} 分数: min={min(scores):.1f}, max={max(scores):.1f}, avg={sum(scores)/len(scores):.1f}")
        
        # 转换为LoRA微调格式
        formatted_data = []
        filtered_count = 0
        truncated_count = 0
        
        for item in reranker_data:
            query = item.get("query", "")
            documents = item.get("documents", [])
            
            # 空查询或无文档，跳过
            if not query or not documents:
                filtered_count += 1
                continue
            
            # 处理文档，截断过长内容
            processed_docs = []
            for doc in documents:
                content = doc.get("page_content", "")
                
                # 跳过空白或极短的文档
                if not content or len(content.strip()) < 10:
                    continue
                
                # 截断过长文档
                if len(content) > max_doc_length:
                    truncated_count += 1
                    content = content[:max_doc_length]
                
                processed_docs.append(content)
            
            # 文档处理后为空，跳过
            if not processed_docs:
                filtered_count += 1
                continue
            
            # 构建输入文本
            formatted_input = f"Question: {query}\n\n"
            for i, doc_content in enumerate(processed_docs):
                formatted_input += f"Document {i+1}:\n{doc_content}\n\n"
            
            # 根据评分模式构建不同的输出格式
            if scoring_mode == "absolute":
                # 绝对评分模式 - 使用scores字段
                scores = item.get("scores", [])
                
                # 确保scores与文档数量匹配
                if len(scores) != len(processed_docs):
                    # 如果不匹配，可能数据有问题，跳过或者做调整
                    # 这里简单处理：如果scores少于文档，补充0分；如果多于文档，截断
                    if len(scores) < len(processed_docs):
                        scores.extend([0.0] * (len(processed_docs) - len(scores)))
                    else:
                        scores = scores[:len(processed_docs)]
                
                # 构建输出文本 - 文档评分格式
                output = ""
                for i in range(len(processed_docs)):
                    # 分数已经在1-5范围内，不需要再乘以5
                    # 确保分数是整数且在1-5范围内
                    score_value = int(round(scores[i]))
                    score_value = max(1, min(5, score_value))
                    output += f"Document {i+1}: {score_value}\n"
                
                instruction = "Rate each document on a scale of 1-5 based on how well it answers the question, where 1 means not relevant at all and 5 means perfectly answering the question with all necessary information."
                formatted_input += "Rate each document from 1 to 5."
                
            else:  # relative mode
                # 相对评分模式 - 选择最佳文档
                best_doc_idx = item.get("best_doc_idx", 0)
                
                # 确保best_doc_idx在合法范围内
                if best_doc_idx >= len(processed_docs):
                    best_doc_idx = 0  # 如果索引超出范围，默认选第一个
                
                output = f"Document {best_doc_idx + 1}"
                instruction = "Identify which document best answers the question. Your response MUST be in the format 'Document X' where X is the document number."
                formatted_input += "Which document best answers the question above?"
            
            # 创建格式化条目
            formatted_item = {
                "instruction": instruction,
                "input": formatted_input,
                "output": output
            }
            
            formatted_data.append(formatted_item)
        
        print(f"Processed {len(reranker_data)} records: {filtered_count} filtered, {truncated_count} truncated, {len(formatted_data)} kept")
        
        # 数据增强
        if len(formatted_data) > 0:
            # 对数据进行增强，如果需要的话
            print("Augmenting data...")
            # 这里可以调用数据增强函数
            # augmented_data = augment_formatted_data(formatted_data)
            # formatted_data = augmented_data
        
        # 如果指定了输出路径，则保存为文件
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(formatted_data)} records to {output_path}")
        
        return formatted_data
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        raise

def filter_long_descriptions(file_path: str, output_path: str, max_length: int = 2048) -> None:
    """
    过滤qa数据中原始描述长度超过指定长度的条目
    
    Args:
        file_path: 原始QA数据文件路径
        output_path: 过滤后的数据保存路径
        max_length: 原始描述的最大长度限制，默认为2048
    
    Returns:
        None
    """
    try:
        # 加载原始QA数据
        with open(file_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        original_count = len(qa_data)
        print(f"原始数据共有 {original_count} 条记录")
        
        # 过滤掉original_description过长的数据
        filtered_data = [item for item in qa_data if len(item.get('original_description', '')) <= max_length]
        
        filtered_count = original_count - len(filtered_data)
        print(f"过滤掉 {filtered_count} 条记录，剩余 {len(filtered_data)} 条记录")
        
        # 保存过滤后的数据
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print(f"过滤后的数据已保存到 {output_path}")
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        raise

def augment_reranker_data(data):
    augmented_data = []
    for item in data:
        # 原始样本
        augmented_data.append(item)

        # 创建变体：调整文档顺序
        if len(item["documents"]) > 2:
            new_item = copy.deepcopy(item)
            docs = new_item["documents"]
            # 保持正确答案位置不变，打乱其他文档
            correct_doc = docs[item["best_doc_idx"]]
            other_docs = [d for i, d in enumerate(docs) if i != item["best_doc_idx"]]
            random.shuffle(other_docs)
            new_docs = other_docs[:item["best_doc_idx"]] + [correct_doc] + other_docs[item["best_doc_idx"]:]
            new_item["documents"] = new_docs
            augmented_data.append(new_item)

        # 创建更多变体...

    return augmented_data



def main():
    convert_reranker_data("../dataset/llm_reranker/absolute_scoring_data.json",
                          "../dataset/llm_reranker/lora_reranker.json")

    
if __name__ == "__main__":
    main()
