import pandas as pd
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
import torch
from tqdm import tqdm
import logging

model_path = '../Models/Qwen/Qwen2.5-7B-Instruct'
train_path = '../dataset/lora_data/prompt1_train.json'
merged_path = "../Models/lora_merged_model"
output_dir="../Models/lora_model/"

"""
{
  "instruction": "回答以下用户问题，仅输出答案。",
  "input": "1+1等于几?",
  "output": "2"
}
"""


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                          trust_remote_code=True)

def process_func(example):
    MAX_LENGTH = 4096  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\nYou are an expert in fault analysis and troubleshooting in the aviation field. "
        f"Based on the given fault isolation procedure and problem, provide a solution.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens

    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    # 将JSON文件转换为CSV文件
    df = pd.read_json(train_path)
    ds = Dataset.from_pandas(df)

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    print(tokenized_id)
    print(tokenizer.decode(tokenized_id[0]['input_ids']))
    print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"]))))

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.enable_input_require_grads()

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=16,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1  # Dropout 比例
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=5,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,

    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        # use_reentrant=False
    )
    trainer.train()
    new_model_directory = merged_path
    merged_model = model.merge_and_unload()
    # 将权重保存为safetensors格式的权重, 且每个权重文件最大不超过2GB(2048MB)
    merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)

if __name__ == '__main__':
    main()
