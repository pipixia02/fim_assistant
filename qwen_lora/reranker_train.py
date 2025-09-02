import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq

model_path = '../Models/Qwen/Qwen2.5-7B-Instruct'
# 更新为重排器训练数据路径
train_path = '../dataset/llm_reranker/lora_reranker.json'
merged_path = "../Models/reranker_lora_merged_model"
output_dir="../Models/reranker_lora_model/"

"""
重排器训练数据格式示例:
{
  "instruction": "Rate each document on a scale of 1-5 based on how well it answers the question.",
  "input": "Question: 什么是云原生?\n\nDocument 1:\n云原生是一种构建和运行应用程序的方法...\n\nDocument 2:\n...",
  "output": "Document 1: 4\nDocument 2: 2\n..."
}
"""


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                          trust_remote_code=True)

def process_func(example):
    MAX_LENGTH = 7000  # 设置足够大的长度以容纳多个文档
    input_ids, attention_mask, labels = [], [], []
    
    # 更新系统提示，适合文档重排序/评分任务
    instruction = tokenizer(
        f"<|im_start|>system\nYou are an expert in document ranking and evaluation. "
        f"Given a question and a set of documents, rate each document based on how well it answers the question "
        f"or select the best document that answers the question most comprehensively.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n", add_special_tokens=False)

    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        print(f"Warning: Sample truncated from {len(input_ids)} to {MAX_LENGTH} tokens")
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    # 加载数据集
    print(f"Loading training data from {train_path}")
    try:
        df = pd.read_json(train_path)
        print(f"Successfully loaded {len(df)} training samples")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return
    
    # # 显示几个示例
    # if len(df) > 0:
    #     print("\nExample training instances:")
    #     for i in range(min(3, len(df))):
    #         print(f"\nExample {i+1}:")
    #         print(f"Instruction: {df.iloc[i]['instruction'][:100]}...")
    #         print(f"Input (truncated): {df.iloc[i]['input'][:100]}...")
    #         print(f"Output: {df.iloc[i]['output']}")
    #         print("-" * 50)
    
    ds = Dataset.from_pandas(df)

    # 分词处理
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    print(f"\nDataset statistics after tokenization:")
    print(f"Number of samples: {len(tokenized_id)}")
    
    # 检查分词后的样本长度
    lengths = [len(x['input_ids']) for x in tokenized_id]
    print(f"Token length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    
    # 打印第一个示例的解码结果
    if len(tokenized_id) > 0:
        print("\nExample of tokenized and decoded data:")
        print("Input:")
        print(tokenizer.decode(tokenized_id[0]['input_ids']))
        print("\nExpected output:")
        print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[0]["labels"]))))

    # 加载模型
    print(f"\nLoading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.enable_input_require_grads()

    # 配置LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=16,  # LoRA秩
        lora_alpha=32,  # LoRA alpha
        lora_dropout=0.1  # Dropout比例
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 训练参数
    # args = TrainingArguments(
    #     output_dir=output_dir,
    #     per_device_train_batch_size=2,  # 减小批次大小以适应更长的序列
    #     gradient_accumulation_steps=8,  # 增加梯度累积步数以补偿较小的批次大小
    #     logging_steps=10,
    #     num_train_epochs=3,
    #     save_steps=50,
    #     learning_rate=5e-5,  # 调整学习率
    #     save_on_each_node=True,
    #     gradient_checkpointing=True,
    #     # 添加评估步骤
    #     evaluation_strategy="steps",
    #     eval_steps=50,
    #     # 添加早停以防止过拟合
    #     load_best_model_at_end=True,
    #     metric_for_best_model="loss",
    #     greater_is_better=False,
    # )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=50,
        num_train_epochs=10,
        save_steps=200,
        learning_rate=5e-5,
        save_on_each_node=True,
        gradient_checkpointing=True,
    )


    # 创建训练器
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    # 开始训练
    print("\nStarting training...")
    trainer.train()
    
    # 合并并保存模型
    print(f"\nMerging and saving model to {merged_path}")
    new_model_directory = merged_path
    merged_model = model.merge_and_unload()
    # 将权重保存为safetensors格式的权重, 且每个权重文件最大不超过2GB(2048MB)
    merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)
    tokenizer.save_pretrained(new_model_directory)
    print("Training complete!")

if __name__ == '__main__':
    main()
