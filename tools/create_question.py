import json
from openai import OpenAI
from typing import List, Dict
import time
import argparse


def load_json_data(file_path: str) -> List[Dict]:
    """Load and parse the JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(type(data))
    data_list = []
    for item in data.keys():
        task = data[item]
        task['task'] = item
        data_list.append(task)

    print(len(data_list))

    return data_list


def filter_descriptions(data: List[Dict]) -> List[Dict]:
    """Filter entries that contain required fields."""
    for item in data:
        if type(item) != dict:
            print(item)

    return [item for item in data if 'Description' in item.keys() and 'Fault Isolation Procedure' in item.keys()]


def generate_qa_with_gpt(item: Dict, client: OpenAI) -> Dict:
    """Generate Q&A pairs using GPT API."""
    try:
        answer_content = item['Fault Isolation Procedure']
        if 'Initial Evaluation' in item:
            answer_content = f"Initial Evaluation: {item['Initial Evaluation']}\nFault Isolation Procedure: {answer_content}"
        
        prompt_1 = f"""Based on the following description and procedure from an aircraft maintenance manual, 
        generate one relevant question and its corresponding answer. 
        The question should be based on the description, and the answer should be summarized from the provided procedure.
        
        Return the result in JSON format with 'question' and 'answer' as keys.
        
        Description: {item['task']+ ' ' +item['Description']}
        Procedure: {answer_content}"""

        prompt_2 = f"""You are an AI assistant specializing in aircraft maintenance. Given the following task description and procedure,  
        generate a precise and relevant question that a maintenance technician might ask.  
        The answer should be summarized concisely from the procedure.  

        Return the result in JSON format with 'question' and 'answer' as keys.  

        Description: {item['task'] + ' ' + item['Description']}  
        Procedure: {answer_content}"""

        prompt_3 = f"""You are an aviation maintenance expert. Based on the aircraft maintenance manual excerpt below, 
        generate a question that requires logical reasoning about the fault and its description. 
        Ensure the question encourages deeper analysis rather than a direct lookup answer.  
        Then, provide a answer summarizing key insights from the fault isolation procedure.  

        Return the result in JSON format with 'question' and 'answer' as keys. 

        Description: {item['task'] + ' ' + item['Description']}  
        Procedure: {answer_content}"""

        prompt_4 = f"""You are an aircraft technician working on fault isolation. 
        A pilot reports an issue related to the following description. 
        Formulate a question that a maintenance technician might ask in a real-world troubleshooting scenario.
        Then, provide a practical answer derived from the fault isolation procedure.

        Return the result in JSON format with 'question' and 'answer' as keys. 

        Description: {item['task'] + ' ' + item['Description']}  
        Procedure: {answer_content}"""

        prompt_5 = f"""Based on the following aircraft maintenance procedure, 
        generate a question that requires knowledge of the step-by-step fault isolation process. 
        The question should focus on "what should be done next" or "which step applies under specific conditions."
        Provide an answer summarizing the relevant procedure.

        Return the result in JSON format with 'question' and 'answer' as keys. 

        Description: {item['task'] + ' ' + item['Description']}  
        Procedure: {answer_content}"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_5}],
            temperature=0.7,
        )
        
        qa_pair = json.loads(response.choices[0].message.content)
        return qa_pair
    
    except Exception as e:
        print(f"Error generating Q&A: {e}")
        return None


def summarize_descriptions_with_llm(item: Dict, client: OpenAI) -> Dict:
    """
    Summarize the original description using LLM API.
    
    Args:
        item: Dictionary containing the original description
        client: OpenAI client for API calls
        
    Returns:
        Dictionary with original and summarized description
    """
    try:
        original_description = item['Description']
        task = item['task']
        
        prompt = f"""You are an aviation maintenance expert with extensive knowledge of aircraft systems and fault isolation procedures. 
        
Provide a technical summary of the following aircraft fault description that:
1. Maintains all critical technical information and fault characteristics
2. Uses precise aviation maintenance terminology
3. Preserves key system identifiers, part numbers, and technical parameters
4. Retains causal relationships and diagnostic indicators
5. Is concise but complete (approximately 30-60% of the original length)

Original Description: {task + ' ' + original_description}

Respond with ONLY the technical summary, without any introductory text, explanations, or conclusions."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  
        )
        
        summarized_description = response.choices[0].message.content.strip()
        
        result = {
            'task': item['task'],
            'original_description': original_description,
            'summarized_description': summarized_description
        }
        
        if 'Fault Isolation Procedure' in item:
            result['original_procedure'] = item['Fault Isolation Procedure']
        
        if 'Initial Evaluation' in item:
            result['original_initial_evaluation'] = item['Initial Evaluation']
            
        return result
    
    except Exception as e:
        print(f"Error summarizing description: {e}")
        return None


def main_summarize():
    """
    Main function to summarize descriptions using LLM API and save to a new JSON file.
    """
    KEY = 'sk-FMw2Yx1uH64II0h6rn69CUY6lHXre48Y8iPVn3cLJBvTt9In'

    client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        api_key=KEY
    )

    data_path = '../dataset/qa_data/fim_data_dict.json'
    output_path = '../dataset/qa_data/summarized_descriptions.json'
    
    data = load_json_data(data_path)
    print(f"Loaded {len(data)} entries from original data")
    
    filtered_data = filter_descriptions(data)
    print(f"Found {len(filtered_data)} entries with descriptions")
    
    summarized_dataset = []
    for i, item in enumerate(filtered_data):
        time.sleep(0.1)
        
        print(f"Summarizing description {i+1}/{len(filtered_data)}: {item['task']}")
        result = summarize_descriptions_with_llm(item, client)
        
        if result:
            summarized_dataset.append(result)
            print(f"Successfully summarized: {item['task']}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summarized_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully created dataset with {len(summarized_dataset)} summarized descriptions")
    print(f"Output saved to: {output_path}")


def main():
    """
    Main function with option to generate QA pairs or summarize descriptions.
    """
    parser = argparse.ArgumentParser(description='Process aircraft maintenance data')
    parser.add_argument('--mode', type=str, choices=['qa', 'summarize'], default='summarize',
                        help='Operation mode: "qa" to generate QA pairs, "summarize" to summarize descriptions')
    
    args = parser.parse_args()
    
    if args.mode == 'summarize':
        print("Running in summarize mode - generating summarized descriptions")
        main_summarize()
    else:
        print("Running in QA mode - generating question-answer pairs")
        KEY = 'sk-FMw2Yx1uH64II0h6rn69CUY6lHXre48Y8iPVn3cLJBvTt9In'

        client = OpenAI(
            base_url='https://xiaoai.plus/v1',
            api_key=KEY
        )

        data_path = '../dataset/qa_data/fim_data_dict.json'
        output_path = '../dataset/qa_data/qa_prompt5.json'
        
        data = load_json_data(data_path)
        print(len(data))
        filtered_data = filter_descriptions(data)
        print(len(filtered_data))
        print(f"Found {len(filtered_data)} entries with descriptions")
        
        qa_dataset = []
        for item in filtered_data:
            time.sleep(0.3)
            
            qa_pair = generate_qa_with_gpt(item, client)
            if qa_pair:
                qa_pair['original_description'] = item['Description']
                qa_pair['original_procedure'] = item['Fault Isolation Procedure']
                qa_pair['task'] = item['task']
                if 'Initial Evaluation' in item:
                    qa_pair['original_initial_evaluation'] = item['Initial Evaluation']
                qa_dataset.append(qa_pair)
                print(qa_pair['task'])
                print(f"Generated Q&A pair {len(qa_dataset)}/{len(filtered_data)}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully created dataset with {len(qa_dataset)} Q&A pairs")

if __name__ == "__main__":
    main()
