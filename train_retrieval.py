import json
import logging
import os

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

# ori_model_path = 'Models/retrieval_models/prompt1_model/large_0.936'
ori_model_path = './Models/BAAI/bge-large-en-v1.5'
output_dir = "Models/retrieval_models/prompt1_model"
train_path = "dataset/retrieval_data/prompt1/summarise/train.json"

dev_path = "dataset/retrieval_data/prompt1/summarise/dev.json"

batch_size = 64
num_epochs = 6
learning_rate = 5e-5
warmup_steps = 400
max_length = 1024


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class RetrievalDataset(Dataset):
    """检索数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """
        初始化数据集
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]

        # 为查询添加特殊前缀（BGE模型推荐）
        query_text = f"Represent this sentence for searching relevant passages: {item['question']}"

        return {
            'query': query_text,
            'positive': item['relevant_doc'],
            'negative': item['irrelevant_doc']
        }


class STRetriever(nn.Module):
    def __init__(self, model_path: str, use_domain_adaptation: bool = True):
        super().__init__()
        # 初始化SentenceTransformer模型
        self.model = SentenceTransformer(model_path)

        # 是否使用领域适应层
        self.use_domain_adaptation = use_domain_adaptation
        print('use domain adaptation:', self.use_domain_adaptation)

        if use_domain_adaptation:
            hidden_size = self.model.get_sentence_embedding_dimension()
            self.query_adapter = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            self.context_adapter = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )

        # 损失函数参数
        self.temperature = 0.06  # 温度参数
        self.margin = 0.3  # margin参数

    def encode_text(self, texts, is_query=False):
        """编码文本，返回归一化的嵌入向量"""
        # SentenceTransformer已经处理了批处理
        embeddings = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)

        # 应用领域适应层（如果启用）
        if self.use_domain_adaptation:
            if is_query:
                # 对于查询，应用query_adapter
                embeddings = embeddings + self.query_adapter(embeddings)  # 残差连接
            else:
                # 对于文档，应用context_adapter
                embeddings = embeddings + self.context_adapter(embeddings)  # 残差连接

        # 再次归一化
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, batch) -> torch.Tensor:
        # 编码查询和文档
        query_embeddings = self.encode_text(batch['query'], is_query=True)
        pos_doc_embeddings = self.encode_text(batch['positive'])
        neg_doc_embeddings = self.encode_text(batch['negative'])

        # 计算相似度
        pos_scores = torch.sum(query_embeddings * pos_doc_embeddings, dim=1) / self.temperature
        neg_scores = torch.matmul(query_embeddings, neg_doc_embeddings.T) / self.temperature

        # 计算带margin的InfoNCE loss
        pos_exp = torch.exp(pos_scores - self.margin)
        # 分母包含所有负样本的得分
        denominator = pos_exp + torch.sum(torch.exp(neg_scores), dim=1)
        # 计算损失
        loss = -torch.log(pos_exp / denominator).mean()

        return loss


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                dev_dataloader: DataLoader,
                output_dir: str,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler,
                device: str) -> float:
    """训练一个epoch，每1/4 epoch验证一次"""
    model.train()
    total_loss = 0
    steps_per_epoch = len(dataloader)
    validation_interval = steps_per_epoch // 4  # 每1/4 epoch验证一次
    best_loss = float('inf')
    with tqdm(dataloader, desc="Training") as pbar:
        for step, batch in enumerate(pbar):
            # 计算损失
            loss = model(batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

            # 每1/4 epoch验证一次
            if step > 0 and step % validation_interval == 0:
                model.eval()
                dev_loss = evaluate(model, dev_dataloader, device)
                logger.info(f"Step {step}/{steps_per_epoch}, Validation loss: {dev_loss:.4f}")

                # 保存最佳模型
                if dev_loss < best_loss and dev_loss < 1.2:
                # if dev_loss < best_loss:
                    best_loss = dev_loss
                    save_model(model, output_dir, best_loss)
                    logger.info(f"New best model saved with loss: {best_loss:.4f}")

                model.train()  # 切回训练模式

    return total_loss / len(dataloader)


def save_model(model, output_dir, loss):
    """保存模型和领域适应层"""
    # 保存模型
    model_path = os.path.join(output_dir, f"sum_model_{loss:.3f}")
    os.makedirs(model_path, exist_ok=True)

    # 保存SentenceTransformer模型
    model.model.save(model_path)
    logger.info(f"Saved SentenceTransformer model to {model_path}")

    # 保存领域适应层
    if model.use_domain_adaptation:
        adapter_path = os.path.join(model_path, "domain_adaptation.pt")
        torch.save({
            'query_adapter': model.query_adapter.state_dict(),
            'context_adapter': model.context_adapter.state_dict()
        }, adapter_path)
        logger.info(f"Saved domain adaptation layers to {adapter_path}")


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    dev_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            loss = model(batch)
            dev_loss += loss.item()
    return dev_loss / len(dataloader)


def main():
    # 设置训练参数


        # 设置随机种子以确保可重复性
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 其他代码不变...
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载分词器和数据集
    tokenizer = AutoTokenizer.from_pretrained(ori_model_path)
    train_dataset = RetrievalDataset(train_path, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = RetrievalDataset(dev_path, tokenizer, max_length)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = STRetriever(model_path=ori_model_path, use_domain_adaptation=True)
    model.to(device)

    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    # 训练循环
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 训练
        train_loss = train_epoch(model, train_dataloader, dev_dataloader,
                                 output_dir, optimizer, scheduler, device)
        logger.info(f"Average training loss: {train_loss:.4f}")

    logger.info("Training completed!")


if __name__ == '__main__':
    main()
