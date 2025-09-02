from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
import torch
from tqdm import tqdm
import logging
import gc

# 配置PyTorch CUDA内存分配器，减少内存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 限制CPU线程数，避免过多线程导致内存压力
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

model_path = '../Models/Qwen/Qwen2.5-7B-Instruct'


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-2.5-7B模型的LoRA微调")
    parser.add_argument("--train_file", type=str, default="./data/train.jsonl",
                      help="训练数据文件")
    parser.add_argument("--validation_file", type=str, default="./data/val.jsonl",
                      help="验证数据文件")
    parser.add_argument("--output_dir", type=str, default="./output",
                      help="输出目录")
    return parser.parse_args()


def print_gpu_memory_usage():
    """打印GPU内存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
            reserved_memory = torch.cuda.memory_reserved(i) / 1024**3
            free_memory = total_memory - reserved_memory
            logger.info(f"GPU {i}: 总内存={total_memory:.2f}GB, 已分配={allocated_memory:.2f}GB, 已预留={reserved_memory:.2f}GB, 可用={free_memory:.2f}GB")


# 使用 Qwen 2.5 的特殊标记格式化对话
def format_qwen_conversation(messages):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += f"<|im_start|>system\n{message['content']}<|im_end|>\n"
        elif message["role"] == "user":
            formatted_text += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
        elif message["role"] == "assistant":
            formatted_text += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"
    return formatted_text


def preprocess_function(examples, tokenizer, max_length):
    conversations = []
    for conv in examples["messages"]:
        # 验证消息格式
        formatted = format_qwen_conversation(conv)
        # 确保没有重复的标记
        conversations.append(formatted)

    tokenized_outputs = tokenizer(
        conversations,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    # 为因果语言模型准备标签
    tokenized_outputs["labels"] = tokenized_outputs["input_ids"].clone()

    return tokenized_outputs


def process_func(example, tokenizer):
    MAX_LENGTH = 4096  # 根据显存调整最大长度

    # 格式化输入
    instruction = tokenizer(
        f"<|im_start|>system\n你是一个航空领域分析故障处理的专家<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False
    )

    # 格式化输出
    response = tokenizer(f"{example['output']}<|im_end|>", add_special_tokens=False)

    # 拼接输入和输出
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 截断超长序列
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def main():
    # 解析命令行参数
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"加载模型和分词器: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 加载模型
    logger.info("以BF16精度加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"  # 在单GPU环境下，明确指定device可能比'auto'更高效
    )
    model.enable_input_require_grads()

    logger.info("配置LoRA参数...")
    lora_config = LoraConfig(
        r=16,  # 降低LoRA秩以减少参数量
        lora_alpha=32,  # 相应降低alpha
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )


    logger.info("应用LoRA适配器到模型...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据集
    logger.info(f"加载训练数据: {args.train_file}")
    logger.info(f"加载验证数据: {args.validation_file}")

    data_files = {
        "train": args.train_file,
        "validation": args.validation_file
    }

    dataset = load_dataset("json", data_files=data_files)

    # 预处理数据
    logger.info("预处理训练数据...")
    processed_datasets = {}
    for split in dataset:
        processed_datasets[split] = dataset[split].map(
            lambda examples: preprocess_function(examples, tokenizer, max_length=4096),
            batched=True,
            remove_columns=dataset[split].column_names
        )

    #print(processed_datasets)
    print(tokenizer.decode(processed_datasets["train"][0]['input_ids']))
    print('-----------------------------------------------')
    print(tokenizer.decode(list(filter(lambda x: x != -100, processed_datasets["train"][1]["labels"]))))

    logger.info("配置训练参数...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,  # 降低评估批次大小
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        gradient_checkpointing=True,  # 在这里启用梯度检查点
    )


    # 创建训练器
    logger.info("创建Trainer...")
    # 使用默认的数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True
    )
    
    # 直接使用Trainer内置的优化器，不再手动设置
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        data_collator=data_collator,
        # 不再手动提供优化器
    )
    

    logger.info("开始LoRA微调...")
    trainer.train()
    
    # 保存模型
    final_model_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    
    logger.info(f"保存最终模型到: {final_model_path}")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info("训练完成!")

if __name__ == '__main__':
    main()
