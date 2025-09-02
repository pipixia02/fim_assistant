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

def search_info(question: str, vector_store: FAISS, top_k: int = 5):
    info_text = ''
    results = vector_store.similarity_search(question, k=top_k)
    
    for result in results:
        info_text += f'''
###description of question:###
{result.page_content}
###Fault Isolation:###
{result.metadata['original_procedure']}\n'''
    
    return info_text

def process_single_question(qa_item, modelname, vectordb):
    """处理单个问题的函数"""
    try:
        question = qa_item['question']
        original_answer = qa_item['answer']

        # 检索相关信息
        information_text = search_info(question, vectordb, top_k=2)

        # 生成prompt并获取回答
        prompt = get_prompt(question, information_text)
        generated_answer = api_retry(modelname, prompt)

        # 返回结果
        return {
            'question': question,
            'original_answer': original_answer,
            'generated_answer': generated_answer
        }
    except Exception as e:
        logger.error(f"Error processing question '{question[:100]}...': {str(e)}")
        return None

def multithread_process(qa_dataset, modelname, vectordb, workers=8):
    """多线程处理问题集"""
    results = []
    futures = []
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务到线程池
        for qa_item in qa_dataset:
            future = executor.submit(
                process_single_question,
                qa_item,
                modelname,
                vectordb
            )
            futures.append(future)
        
        # 使用tqdm显示进度
        for future in tqdm(futures, desc="Processing questions"):
            try:
                result = future.result(timeout=300)  # 5分钟超时
                if result is not None:
                    results.append(result)
                    # 每完成一个问题就保存一次中间结果

            except Exception as e:
                logger.error(f"Error getting result from future: {str(e)}")
                continue
            
            # 添加短暂延迟避免API限制
            time.sleep(0.2)
    
    return results

def main():
    modelname = MODEL_NAME
    prompt = 'who are you'
    generated_answer = api_retry(modelname, prompt)
    print(generated_answer)
    qa_dataset = load_qa_dataset(qa_data_path)
    # 加载向量数据库
    vectordb = load_vector_store(
        model_path=retrieval_model_path,
        faiss_index_path=faiss_index_path,
        use_custom_model=True
    )

    try:
        results = multithread_process(qa_dataset, MODEL_NAME, vectordb, workers=16)
        print(f"Successfully processed {len(results)} questions")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

    # 保存最终结果
    output_path = f"generated_answers_{str(time.localtime(time.time()))}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()