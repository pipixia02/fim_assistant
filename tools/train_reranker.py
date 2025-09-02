import os

from sentence_transformers import CrossEncoder

from tqdm import tqdm
from transformers import AutoTokenizer,get_cosine_schedule_with_warmup
from typing import List, Dict, Optional, Tuple
import json
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import numpy as np


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, scores, labels):
        # 将标签转换为-1和1
        labels = 2 * labels - 1
        # 计算损失
        loss = torch.mean((1 - labels) * torch.pow(scores, 2) +  # 负样本的损失
                          labels * torch.pow(torch.clamp(self.margin - scores, min=0.0), 2))  # 正样本的损失
        return loss


class MarginRankingLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_scores, neg_scores):
        # 计算排序损失
        loss = torch.mean(torch.clamp(self.margin - pos_scores + neg_scores, min=0.0))
        return loss


class TextPairDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.features[idx]['input_ids'],
            'attention_mask': self.features[idx]['attention_mask'],
        }
        if self.features[idx]['token_type_ids'] is not None:
            item['token_type_ids'] = self.features[idx]['token_type_ids']
        return item, self.labels[idx]


def prepare_training_data(train_data: List[Dict], tokenizer_name: str, max_length: int = 512) -> Dataset:
    """准备训练数据，使用pair格式"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    if not train_data:
        logger.error("Empty training data")
        raise ValueError("Empty training data")

    train_features = []
    train_labels = []

    for item in train_data:
        query = item['question']
        positive_doc = item['relevant_doc']

        # 使用pair格式处理正样本
        encoded = tokenizer(
            query,
            positive_doc,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        train_features.append({
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'token_type_ids': encoded['token_type_ids'].squeeze(0) if 'token_type_ids' in encoded else None
        })
        train_labels.append(1.0)

        # 处理难负样本
        for neg_doc in item['irrelevant_doc']:
            encoded = tokenizer(
                query,
                neg_doc,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            train_features.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'token_type_ids': encoded['token_type_ids'].squeeze(0) if 'token_type_ids' in encoded else None
            })
            train_labels.append(0.0)

    train_labels = torch.tensor(train_labels, dtype=torch.float)
    return TextPairDataset(train_features, train_labels)


class RerankerTrainer:
    def __init__(
            self,
            model: CrossEncoder,
            batch_size: int = 32,
            num_epochs: int = 3,
            learning_rate: float = 1e-5,
            save_path: str = './reranker_model',
            margin: float = 0.5
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.base_model = self.model.model
        self.base_model.to(self.device)

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.save_path = save_path

        # 使用多个损失函数
        self.contrastive_loss = ContrastiveLoss(margin=margin).to(self.device)
        self.ranking_loss = MarginRankingLoss(margin=margin).to(self.device)
        self.bce_loss = torch.nn.BCEWithLogitsLoss().to(self.device)

        # 使用带权重衰减的优化器
        self.optimizer = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01  # 添加L2正则化
        )

        logger.info(f"Initialized trainer on device: {self.device}")

    def train_step(self, batch, labels):
        """单步训练"""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.base_model(**batch)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        scores = logits.squeeze()

        # 计算BCE损失
        bce_loss = self.bce_loss(scores, labels)

        # 计算对比损失
        contrastive_loss = self.contrastive_loss(scores, labels)

        # 分离正负样本并确保维度匹配
        pos_mask = labels == 1
        neg_mask = labels == 0

        if torch.any(pos_mask) and torch.any(neg_mask):
            pos_scores = scores[pos_mask]
            neg_scores = scores[neg_mask]

            # 确保正负样本数量相同
            min_samples = min(pos_scores.size(0), neg_scores.size(0))
            pos_scores = pos_scores[:min_samples]
            neg_scores = neg_scores[:min_samples]

            ranking_loss = self.ranking_loss(pos_scores, neg_scores)
        else:
            ranking_loss = torch.tensor(0.0).to(self.device)

        # 组合损失
        total_loss = bce_loss + contrastive_loss # + ranking_loss

        return total_loss, {
            'bce_loss': bce_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'ranking_loss': ranking_loss.item()
        }

    def _training_callback(self, step: int, loss_dict: Dict[str, float], val_dataloader: DataLoader, epoch: int) -> \
    Tuple[int, float]:
        """训练过程中的回调函数，用于定期评估和保存模型"""
        if val_dataloader:
            self.base_model.eval()
            metrics = self.evaluate(val_dataloader)
            self.base_model.train()

            logger.info(
                f"Epoch: {epoch + 1}, Step: {step}, "
                f"Training Loss: {loss_dict['bce_loss']:.4f}, "
                f"Validation Loss: {metrics['loss']:.4f}, "
                f"Accuracy: {metrics['accuracy']:.4f}"
            )

            return step, metrics['accuracy']


    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.base_model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch, labels in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = labels.to(self.device)

                outputs = self.base_model(**batch)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                scores = torch.sigmoid(logits.squeeze())

                loss = self.bce_loss(logits.squeeze(), labels)
                total_loss += loss.item()

                all_preds.extend((scores > 0.5).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': np.mean(np.array(all_preds) == np.array(all_labels))
        }

        return metrics


    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None,
    ):
        """训练模型"""
        logger.info("Starting training...")

        # 学习率预热和衰减
        num_training_steps = len(train_dataloader) * self.num_epochs
        warmup_steps = int(num_training_steps * 0.1)
        scheduler = get_cosine_schedule_with_warmup(  # 使用余弦退火
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        best_acc = 0
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            # 训练阶段
            self.base_model.train()
            total_loss = 0



            with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}") as pbar:
                for step, (batch, labels) in enumerate(pbar):
                    loss, loss_dict = self.train_step(batch, labels)
                    loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)

                    self.optimizer.step()
                    scheduler.step()

                    total_loss += loss.item()
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'bce': f"{loss_dict['bce_loss']:.4f}",
                        'cont': f"{loss_dict['contrastive_loss']:.4f}",
                        'rank': f"{loss_dict['ranking_loss']:.4f}"
                    })

                    # 定期评估和保存
                    if step > 0 and step % (len(train_dataloader) // 5) == 0:
                        step, acc = self._training_callback(step, loss_dict, val_dataloader, epoch)
                        if acc > best_acc:
                            # 保存最佳模型
                            self.save_model(acc)
                            logger.info(f"New best model saved with accuracy: {acc:.4f}")
                            best_acc = acc
                        #     patience_counter = 0
                        # else:
                        #     patience_counter += 1

                    # # 早停
                    # if patience_counter >= early_stopping_patience:
                    #     logger.info("Early stopping triggered")
                    #     return

    def save_model(self, acc):
        """保存模型"""
        save_path = os.path.join(self.save_path, f"best_model_{acc:.3f}")
        os.makedirs(save_path, exist_ok=True)
        self.model.save(save_path)
        logger.info(f"Saved model to {save_path}")



def main():
    # 配置日志
    logger.add("logs/training_{time}.log")

    # 配置
    config = {
        'model_name': "./Models/retrieval_models/best_model0.784",
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 1e-5,
        'save_path': './Models/reranker_model',
        'train_data_path': './dataset/reranker/train_reranker_data.json',
        'val_ratio': 0.15
    }

    # 1. 初始化模型
    model = CrossEncoder(
        model_name=config['model_name'],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # 2. 加载和准备数据
    with open(config['train_data_path'], 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    dataset = prepare_training_data(train_data, config['model_name'])
    # 划分训练集和验证集
    train_size = int((1 - config['val_ratio']) * len(dataset))
    val_size = len(dataset) - train_size
    print('train_size:',train_size, 'val_size:', val_size)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size']
    )

    # 3. 初始化训练器并训练
    trainer = RerankerTrainer(
        model=model,
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        save_path=config['save_path']
    )

    trainer.train(train_dataloader, val_dataloader)
    logger.info("Training completed")


if __name__ == '__main__':
    main()