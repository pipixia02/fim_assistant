import json
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_community.vectorstores import FAISS
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from create_faissdb import load_qa_dataset, load_vector_store

MODEL_NAME = 'Qwen2.5-7B-Instruct-local'
model_path = "Models/Qwen/Qwen2.5-7B-Instruct"
faiss_index_path = 'database/faiss_index_v2'
retrieval_model_path = 'Models/retrieval_models/best_model0.6'
qa_data_path = 'dataset/qa_dataset.json'

def call_qwen_api(model_name, query):
    try:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="sk-xxx",  # 随便填写，只是为了通过接口参数校验
        )
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role':'system','content':'You are a professional aviation domain expert specializing in answering questions about the aviation industry. '},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        return None

def api_retry(model_name, query):
    max_retries = 5
    retry_delay = 60  # in seconds
    attempts = 0
    while attempts < max_retries:
        try:
            return call_qwen_api(model_name, query)
        except Exception as e:
            attempts += 1
            if attempts < max_retries:
                logger.warning(f"Attempt {attempts} failed for text: {query}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"All {max_retries} attempts failed for text: {query}. Error: {e}")
                raise

def get_prompt(question, context):
    prompt = f"""You are a professional aviation domain expert specializing in answering questions about the aviation industry. Your responsibilities include providing accurate, detailed, and context-specific answers based on the retrieved external knowledge. The retrieved information will be appended as context before each question. Your tasks are:

1. Use the retrieved information as the primary source for your answers.
2. If the retrieved information is insufficient or unrelated, supplement your response with general aviation expertise.
3. Keep your answers concise yet informative, using a professional tone.
4. Acknowledge when a question is outside the aviation domain or when information is unavailable.

**Contexts**:
{context}

Now, use the provided context to answer the user's question as accurately as possible. If no context is available or relevant, rely on your aviation expertise.

### Question:
{question}
"""
    return prompt
