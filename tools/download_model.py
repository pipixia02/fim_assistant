
from modelscope import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

# 模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-large-en-v1.5', cache_dir='../Models')


#模型下载
# from modelscope import snapshot_download
# model_dir = snapshot_download('bert-large-uncased', cache_dir='./')

# from sentence_transformers import CrossEncoder
#
# # 加载 CrossEncoder
# model_name = "../Models/bert-large-uncased"
# cross_encoder = CrossEncoder(model_name)
#
# # 访问模型参数
# print(cross_encoder)
# parameters = cross_encoder.model.parameters()
# for param in parameters:
#     print(param.size())  # 打印每个参数的形状
#     break

# def call_qwen_api(model_path, query):
#     try:
#         client = OpenAI(
#             base_url="http://localhost:8000/v1",
#             api_key="sk-xxx",
#         )
#         print('creating chat...  query:', query)
#         completion = client.chat.completions.create(
#             model=model_path,
#             messages=[
#                 {'role': 'system',
#                  'content': 'You are a professional aviation domain expert specializing in answering questions about the aviation industry. '},
#                 {"role": "user", "content": query}
#             ],
#             temperature=0.7,  # 可以添加温度参数控制输出的随机性
#             max_tokens=1024   # 可以控制输出的最大长度
#         )
#         return completion.choices[0].message.content
#     except Exception as e:
#         print(f"Error calling API: {str(e)}")
#         return None
#
# if __name__ == '__main__':
#     model_path = "../Models/Qwen/Qwen2.5-7B-Instruct"
#
#     # 确保本地服务已启动
#     try:
#         response = call_qwen_api(model_path, 'What is the largest language model?')
#         if response:
#             print("Response:", response)
#         else:
#             print("Failed to get response from the model")
#     except Exception as e:
#         print(f"Error: {str(e)}")