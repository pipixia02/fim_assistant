import json

from langchain_community.document_loaders import PyPDFLoader
import re
from PyPDF2 import PdfReader

import os
import re
from PyPDF2 import PdfReader

folder_path = 'D:/plane/pdf/'


# # 定义正则模式：匹配任务标题、子标题和任务结束标志
# task_start_pattern = re.compile(r"(\d+\.\s*[\w\s-]+)\s*-\s*Fault Isolation", re.IGNORECASE)  # 任务标题
#
# subheading_pattern = re.compile(
#     r"(A\.\s*Description|B\.\s*Possible Causes|C\.\s*Circuit Breakers|D\.\s*Related Data|E\.\s*Initial Evaluation|F\.\s*Fault Isolation Procedure)(.*?)((?=^[A-F]\.\s)|\Z)",
#     re.DOTALL | re.MULTILINE)  # 子标题
# end_task_pattern = re.compile(r"------------------------- END OF TASK ---------------------------",
#                               re.IGNORECASE)  # 任务结束标记
#
# # 列表存储所有任务，每个任务为一个字典
# tasks_list = []
#
# # 合并PDF文本
# all_text = ""
# for page in reader.pages:
#     all_text += page.extract_text() + "\n"
#
# # 分割任务文本，依据任务结束标志
# task_texts = re.split(end_task_pattern, all_text)
#
# # 遍历每个任务文本片段
# for task_text in task_texts:
#     # 查找任务标题
#     task_match = task_start_pattern.search(task_text)
#     if task_match:
#         task_dict = {}
#         task_dict[task_match.group().strip()] = {}  # 提取任务标题
#
#         # 提取子标题和内容
#         subheading_matches = subheading_pattern.findall(task_text)
#         for match in subheading_matches:
#             key = match[0].replace('.', '').strip()  # 清理键格式
#             value = match[1].strip()
#             task_dict[key] = value
#
#         # 将每个任务字典添加到任务列表中
#         tasks_list.append(task_dict)
#
# # 输出部分结果验证
# print(tasks_list[:3])  # 显示前三个任务以供验证


task_start_pattern = re.compile(r"(\d+\.\s*[\w\s-]+)\s*-\s*Fault Isolation", re.IGNORECASE)  # 匹配任务标题
task_end_marker = "\n------------------------- END OF TASK ---------------------------\n"  # 任务终止标记
page_info_pattern = re.compile(r"(Page\s*\d+|D\d{6,}-XIA\s*[\w\s/]+)")  # 去除页面编号和日期信息
subheading_pattern = re.compile(
    r"([A-K]\.\s*(?:Description|Possible Causes|Circuit Breakers|Related Data|Initial Evaluation|"
    r"Tools and Equipment|Fault Isolation(?: Procedure)?|Repair Confirmation))\s*(.*?)(?=^[A-K]\.\s|\Z)",
    re.S | re.M
)

tasks = []
current_task = {}


def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            print(file_path)
            process_pdf(file_path)

def process_pdf_sum(pdf_path):
    sum = 0
    current_task_name = None
    current_task_content = []

    # 加载PDF文件
    # pdf_path = './pdf/21___047.PDF'
    reader = PdfReader(pdf_path)

    # 遍历PDF文件的每一页
    for page in reader.pages:
        text = page.extract_text()
        if text:
            # 分割任务块
            for line in text.splitlines():
                # 检查是否为任务的开始
                if line.strip() == task_end_marker.strip():
                    sum += 1

    print(sum)

def process_pdf(pdf_path):
    current_task_name = None
    current_task_content = []

    # 加载PDF文件
    # pdf_path = './pdf/21___047.PDF'
    reader = PdfReader(pdf_path)

    # 遍历PDF文件的每一页
    for page in reader.pages:
        text = page.extract_text()
        if text:
            # 分割任务块
            for line in text.splitlines():
                # 检查是否为任务的开始
                if match := task_start_pattern.match(line):

                    # 如果有正在积累的任务内容，将其作为字典加入任务列表并重置
                    if current_task_name:
                        task_content_cleaned = "\n".join(current_task_content)
                        # 去除无用信息
                        task_content_cleaned = page_info_pattern.sub("", task_content_cleaned).strip()
                        current_task[current_task_name] = task_content_cleaned
                        tasks.append(current_task)

                    # 开始新的任务
                    current_task_name = match.group(1).strip()
                    current_task_name = current_task_name[5:]
                    current_task = {}
                    current_task_content = [line]

                elif line.strip() == task_end_marker.strip():
                    # 碰到任务的终止标记，保存当前任务到列表
                    if current_task_name:
                        task_content_cleaned = "\n".join(current_task_content)
                        # 去除无用信息
                        task_content_cleaned = page_info_pattern.sub("", task_content_cleaned).strip()
                        current_task[current_task_name] = task_content_cleaned
                        tasks.append(current_task)
                        current_task_name = None
                        current_task_content = []
                else:
                    # 否则继续积累任务内容
                    if current_task_name:
                        current_task_content.append(line)


if __name__ == '__main__':

    # subheading_pattern = re.compile(
    #     r"(A\.\s*Description|B\.\s*Possible Causes|C\.\s*Circuit Breakers|D\.\s*Related Data|E\.\s*Initial Evaluation|F\.\s*Fault Isolation Procedure)(.*?)((?=^[A-F]\.\s)|\Z)",
    #     re.DOTALL | re.MULTILINE)  # 子标题
    #
    # subheading_pattern = re.compile(
    #     r"(A\.\s*Description|B\.\s*Possible Causes|C\.\s*Circuit Breakers|D\.\s*Related Data|E\.\s*(Initial Evaluation|Tools and Equipment)|F\.\s*(Fault Isolation Procedure|Initial Evaluation)|"
    #     r"G\.\s*(Fault Isolation|Repair Confirmation)|H\.\s*(Fault Isolation|Repair Confirmation))(.*?)((?=^[A-H]\.\s)|\Z)",
    #     re.DOTALL | re.MULTILINE
    # )

    process_folder(folder_path)

    # 输出示例任务

    task_dicts = {}
    for task in tasks:
        for key in task.keys():
            subheading_matches = subheading_pattern.findall(task[key])
            sub_task = {}
            for match in subheading_matches:
                sub_key = match[0].strip()[3:]  # 清理键格式
                sub_value = match[1].strip()
                if sub_key not in sub_task.keys():
                    sub_task[sub_key] = sub_value
                else:
                    sub_task[sub_key] = sub_task[sub_key] + sub_value

            task[key] = sub_task

            task_dicts[key] = sub_task


    print(len(tasks))  # 显示前5个任务

    with open('../dataset/qa_data/fim_data_dict.json', 'w', encoding='utf8') as file_obj:

        json.dump(task_dicts, file_obj, indent=4)


