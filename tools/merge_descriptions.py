#!/usr/bin/env python3
import json
import os
from typing import List, Dict

def load_json_data(file_path: str) -> List[Dict]:
    """Load and parse the JSON file."""
    print(f"Loading data from {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} items from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        raise

def save_json_data(file_path: str, data: List[Dict]) -> None:
    """Save data to a JSON file."""
    print(f"Saving {len(data)} items to {file_path}")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved data to {file_path}")
    except Exception as e:
        print(f"Error saving to file {file_path}: {e}")
        raise

def merge_descriptions():
    """
    Merge summarized descriptions into qa dataset:
    1. Load summarized descriptions and original QA data
    2. Create a lookup dictionary for summarized descriptions by task
    3. Add summarized_description to corresponding QA entries
    4. Remove entries without summarized_description
    5. Save the updated dataset
    """
    # File paths
    summarized_path = '../dataset/qa_data/summarized_descriptions.json'
    qa_path = '../dataset/qa_data/qa_prompt1.json'
    output_path = '../dataset/qa_data/qa_prompt1_with_summarise.json'
    
    # Load data
    summarized_data = load_json_data(summarized_path)
    qa_data = load_json_data(qa_path)
    
    # Create lookup dictionary for summarized descriptions by task
    summary_lookup = {}
    for item in summarized_data:
        if 'task' in item and 'summarized_description' in item:
            summary_lookup[item['task']] = item['summarized_description']
    
    print(f"Created lookup dictionary with {len(summary_lookup)} summarized descriptions")
    
    # Add summarized descriptions to QA entries
    updated_qa_data = []
    for item in qa_data:
        if 'task' in item and item['task'] in summary_lookup:
            # Add summarized description to this item
            item['summarized_description'] = summary_lookup[item['task']]
            updated_qa_data.append(item)
        else:
            print(f"Skipping item (no matching summarized description): {item.get('task', 'No task ID')}")
    
    print(f"Original QA entries: {len(qa_data)}")
    print(f"Updated QA entries with summarized descriptions: {len(updated_qa_data)}")
    print(f"Removed entries: {len(qa_data) - len(updated_qa_data)}")
    
    # Save updated dataset
    save_json_data(output_path, updated_qa_data)
    
    # Optionally, also update the original file (uncomment if needed)
    # save_json_data(qa_path, updated_qa_data)
    
    print(f"Process completed. Updated dataset saved to {output_path}")

if __name__ == "__main__":
    merge_descriptions()
