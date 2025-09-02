import json
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from retrieval.retrieval_model import CustomSentenceTransformerEmbeddings
from langchain_core.documents import Document
from typing import List, Dict
import os
from loguru import logger

# 配置参数
faiss_index_path = 'database/faiss_prompt1_sum'
qa_dataset_path = 'dataset/qa_data/qa_prompt1_with_summarise.json'
model_path = "Models/retrieval_models/prompt1_model/sum_model_0.759"  # 使用训练好的模型


def load_descriptions_from_qa_dataset(file_path: str) -> List[Document]:
    """从QA数据集加载original_description并转换为Document格式"""
    with open(file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    documents = []
    for item in qa_data:
        # 创建Document对象，包含文本内容和可选的元数据
        if 'original_initial_evaluation' not in item.keys():
            original_initial_evaluation = None
        else:
            original_initial_evaluation = item['original_initial_evaluation']
        doc = Document(
            page_content=item['task'] + ' ' +item['original_description'],
            # page_content=item['task'] + ' ' + item['summarized_description'],
            metadata={
                'question': item['question'],
                'answer': item['answer'],
                'task':item['task'],
                'original_procedure': item['original_procedure'],
                'original_initial_evaluation': original_initial_evaluation
            }
        )
        documents.append(doc)

    print(f"Loaded {len(documents)} documents from {file_path}")
    return documents



def create_vector_store(documents: List[Document],
                        model_path: str,
                        faiss_index_path: str,
                        use_custom_model: bool = True) -> FAISS:

    # for doc in documents:
    #     print(doc.page_content)
    #     print(doc.metadata.keys())
    """创建向量存储"""
    if use_custom_model:
        # 使用我们训练好的自定义模型
        embed_model = CustomSentenceTransformerEmbeddings(
            model_path=model_path,
            use_domain_adaptation=True,
            pooling_strategy='mean'
        )
    else:
        # 使用原始的BGE模型
        embed_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # BGE 模型推荐设置
        )

    # 创建或加载FAISS索引
    if not os.path.exists(faiss_index_path):
        vectordb = FAISS.from_documents(
            documents=documents,
            embedding=embed_model
        )
        vectordb.save_local(faiss_index_path)
        print(f"Created FAISS index at {faiss_index_path}")
    else:

        vectordb = FAISS.load_local(faiss_index_path, embed_model, allow_dangerous_deserialization=True)
        print(f"Loading existing FAISS index from {faiss_index_path}")

    return vectordb


def load_vector_store(model_path: str,
                      faiss_index_path: str,
                      use_custom_model: bool = False) -> FAISS:
    """创建向量存储"""
    if use_custom_model:
        # 使用我们训练好的自定义模型
        embed_model = CustomSentenceTransformerEmbeddings(
            model_path=model_path,
            use_domain_adaptation=True,
            pooling_strategy='mean'
        )
        logger.info(f"Successfully loaded custom model from {model_path}")
    else:
        # 使用原始的BGE模型
        embed_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # BGE 模型推荐设置
        )
        logger.info(f"Successfully loaded BGE model from {model_path}")
    try:
        # 加载现有的 FAISS 索引
        #vectordb = FAISS.load_local(faiss_index_path, embed_model)
        vectordb = FAISS.load_local(faiss_index_path, embed_model, allow_dangerous_deserialization=True)

        logger.info(f"Successfully loaded FAISS index from disk. use_custom_model：{use_custom_model}")
    except Exception as e:

        logger.warning(f"Could not load FAISS index from {faiss_index_path}: {e}. Creating a new index.")
        # 如果无法加载现有索引，则创建一个新的
        # vectordb = FAISS.from_documents([], embed_model)
        # vectordb.save_local(faiss_index_path)
        raise e
    return vectordb


def load_qa_dataset(file_path: str) -> List[Dict]:

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        logger.info(f"Successfully loaded {len(data)} QA pairs from {file_path}")
        return data


def main():
    # 加载文档
    documents = load_descriptions_from_qa_dataset(qa_dataset_path)
    # 创建向量存储
    # 使用训练好的模型
    vectordb = create_vector_store(
        documents=documents,
        model_path=model_path,
        faiss_index_path=faiss_index_path,
        use_custom_model=True  # 设置为False使用原始BGE模型
    )

    # 测试查询
    test_query = "What is the purpose of FIM?"
    results = vectordb.similarity_search(test_query, k=2)
    print("\nTest Query Results:")
    for doc in results:
        print(f"\nContent: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")


if __name__ == '__main__':
    main()
