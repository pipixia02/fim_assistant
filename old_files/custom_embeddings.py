import torch
from typing import List, Optional, Union
from langchain.embeddings.base import Embeddings
from transformers import AutoModel, AutoTokenizer
import os

class DomainAdaptationLayer(torch.nn.Module):
    """领域适应层"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)  # 残差连接


class CustomBGEEmbeddings(Embeddings):
    def __init__(self,model_path: str, use_domain_adaptation: bool = True,pooling_strategy: str = 'mean', device: Optional[str] = None,):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path)
        # 加载领域适应层
        if use_domain_adaptation:
            adapter_path = os.path.join(model_path, "domain_adaptation.pt")
            if os.path.exists(adapter_path):
                # 初始化领域适应层
                self.context_adapter = DomainAdaptationLayer(self.model.config.hidden_size)
                # 加载状态字典
                if torch.cuda.is_available():
                    adapters = torch.load(adapter_path, weights_only=False)
                else:
                    adapters = torch.load(adapter_path, map_location=torch.device('cpu'))
                # 加载context_adapter的状态
                self.context_adapter.load_state_dict(adapters['context_adapter'])
                self.use_domain_adaptation = True
            else:
                print(f"Warning: Domain adaptation file not found at {adapter_path}")
                self.use_domain_adaptation = False
        else:
            self.use_domain_adaptation = False
        self.pooling_strategy = pooling_strategy

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self.model.to(device)
        if self.use_domain_adaptation:
            self.context_adapter = self.context_adapter.to(device)

    def mean_pooling(self, token_embeddings: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        """计算平均池化"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_text(self, text: str) -> List[float]:
        # 添加前缀
        text = f"Represent this sentence for searching relevant passages: {text}"
        text = text
        # 编码文本
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

            # 根据池化策略选择文本表示
            if self.pooling_strategy == 'cls':
                embeddings = outputs.last_hidden_state[:, 0]
            else:  # mean
                embeddings = self.mean_pooling(
                    outputs.last_hidden_state,
                    inputs['attention_mask']
                )

            # 应用领域适应层（如果启用）
            if self.use_domain_adaptation:
                embeddings = self.context_adapter(embeddings)

            # 归一化
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings[0].cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the custom model"""
        return [self.encode_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the custom model"""
        return self.encode_text(text)


