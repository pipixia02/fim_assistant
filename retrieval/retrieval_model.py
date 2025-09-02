import torch
from typing import List, Optional, Union
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import os

class CustomSentenceTransformerEmbeddings(Embeddings):
    def __init__(
        self,
        model_path: str,
        use_domain_adaptation: bool = False,
        pooling_strategy: str = 'mean',
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        
        # 设置设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # 初始化SentenceTransformer模型
        self.model = SentenceTransformer(model_path, device=device)
        # 注意：SentenceTransformer已经内置了池化策略，这里保留参数是为了保持接口一致
        self.pooling_strategy = pooling_strategy
        self.use_domain_adaptation = False
        # 如果需要领域适应，加载领域适应层
        if use_domain_adaptation:
            adapter_path = os.path.join(model_path, "domain_adaptation.pt")
            if os.path.exists(adapter_path):
                # 初始化领域适应层
                hidden_size = self.model.get_sentence_embedding_dimension()
                self.query_adapter = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, hidden_size)
                )
                self.context_adapter = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, hidden_size)
                )
                # 加载状态字典
                if torch.cuda.is_available():
                    adapters = torch.load(adapter_path, weights_only=True)
                else:
                    adapters = torch.load(adapter_path,weights_only=True, map_location=torch.device('cpu'))
                
                # 加载adapter的状态
                self.query_adapter.load_state_dict(adapters['query_adapter'])
                self.context_adapter.load_state_dict(adapters['context_adapter'])
                
                # 将adapter移到正确的设备上
                self.query_adapter = self.query_adapter.to(device)
                self.context_adapter = self.context_adapter.to(device)
                
                self.use_domain_adaptation = True
            else:
                print(f"Warning: Domain adaptation file not found at {adapter_path}")
                self.use_domain_adaptation = False

    def encode_text(self, text: str, is_query: bool = True) -> List[float]:
        # 添加前缀，与原实现保持一致
        text = f"Represent this sentence for searching relevant passages: {text}"
        
        # 使用SentenceTransformer编码文本
        with torch.no_grad():
            embeddings = self.model.encode(text, convert_to_tensor=True,normalize_embeddings=True,show_progress_bar=False)
            
            # 应用领域适应层（如果启用）
            if self.use_domain_adaptation:
                if is_query:
                    # 对于查询，应用query_adapter
                    embeddings = embeddings + self.query_adapter(embeddings)  # 残差连接
                else:
                    # 对于文档，应用context_adapter
                    embeddings = embeddings + self.context_adapter(embeddings)  # 残差连接
            
            # 归一化
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=0)
            
        return embeddings.cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the SentenceTransformer model"""
        # 添加前缀
        texts = [f"Represent this sentence for searching relevant passages: {text}" for text in texts]
        
        # 批量编码
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            
            # 应用领域适应层（如果启用）
            if self.use_domain_adaptation:
                # 对于文档，应用context_adapter
                embeddings = embeddings + self.context_adapter(embeddings)  # 残差连接
            
            # 归一化
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings.cpu().tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the SentenceTransformer model"""
        return self.encode_text(text, is_query=True)

if __name__ == '__main__':
    model_path = './Models/st_model/best_model_0.690'
    model = CustomSentenceTransformerEmbeddings(model_path, use_domain_adaptation=True)
    print('load model success')
