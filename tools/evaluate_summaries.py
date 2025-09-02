#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import random

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_json_data(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def calculate_compression_ratio(original, summary):
    """Calculate compression ratio (summary length / original length)."""
    if len(original) == 0:
        return 0
    return len(summary) / len(original)

def calculate_cosine_similarity(text1, text2):
    """Calculate cosine similarity between two texts using TF-IDF."""
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]

def calculate_word_overlap_ratio(text1, text2):
    """Calculate the ratio of words from text1 that appear in text2."""
    words1 = set(word_tokenize(text1.lower()))
    words2 = set(word_tokenize(text2.lower()))
    
    if len(words1) == 0:
        return 0
    
    intersection = words1.intersection(words2)
    return len(intersection) / len(words1)

def count_technical_terms(text, min_length=6):
    """Estimate the number of technical terms by counting longer words."""
    # This is a simple heuristic - words longer than min_length are more likely to be technical terms
    words = word_tokenize(text.lower())
    return sum(1 for word in words if len(word) >= min_length)

def evaluate_summaries():
    """Evaluate the quality of summarized descriptions."""
    # Load data
    data_path = '../dataset/qa_data/summarized_descriptions.json'
    data = load_json_data(data_path)
    
    # Initialize metrics
    compression_ratios = []
    similarity_scores = []
    word_overlap_ratios = []
    tech_term_retention = []
    
    # Sample for detailed analysis
    sample_indices = random.sample(range(len(data)), min(5, len(data)))
    samples = []
    
    # Process each item
    for i, item in enumerate(data):
        if 'original_description' in item and 'summarized_description' in item:
            original = item['task'] + ' ' + item['original_description']
            summary = item['summarized_description']
            
            # Calculate metrics
            comp_ratio = calculate_compression_ratio(original, summary)
            sim_score = calculate_cosine_similarity(original, summary)
            overlap = calculate_word_overlap_ratio(original, summary)
            
            # Technical terms
            original_tech_terms = count_technical_terms(original)
            summary_tech_terms = count_technical_terms(summary)
            tech_retention = (summary_tech_terms / original_tech_terms) if original_tech_terms > 0 else 0
            
            # Store metrics
            compression_ratios.append(comp_ratio)
            similarity_scores.append(sim_score)
            word_overlap_ratios.append(overlap)
            tech_term_retention.append(tech_retention)
            
            # Store samples for detailed analysis
            if i in sample_indices:
                samples.append({
                    'task': item['task'],
                    'original': original,
                    'summary': summary,
                    'comp_ratio': comp_ratio,
                    'similarity': sim_score,
                    'word_overlap': overlap,
                    'tech_retention': tech_retention
                })
    
    # Calculate statistics
    stats = {
        'compression_ratio': {
            'mean': np.mean(compression_ratios),
            'std': np.std(compression_ratios),
            'min': np.min(compression_ratios),
            'max': np.max(compression_ratios)
        },
        'similarity_score': {
            'mean': np.mean(similarity_scores),
            'std': np.std(similarity_scores),
            'min': np.min(similarity_scores),
            'max': np.max(similarity_scores)
        },
        'word_overlap': {
            'mean': np.mean(word_overlap_ratios),
            'std': np.std(word_overlap_ratios),
            'min': np.min(word_overlap_ratios),
            'max': np.max(word_overlap_ratios)
        },
        'tech_term_retention': {
            'mean': np.mean(tech_term_retention),
            'std': np.std(tech_term_retention),
            'min': np.min(tech_term_retention),
            'max': np.max(tech_term_retention)
        }
    }
    
    # Create histograms
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(compression_ratios, bins=20, alpha=0.7)
    plt.title('Compression Ratio Distribution')
    plt.xlabel('Summary Length / Original Length')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 2)
    plt.hist(similarity_scores, bins=20, alpha=0.7)
    plt.title('Cosine Similarity Distribution')
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 3)
    plt.hist(word_overlap_ratios, bins=20, alpha=0.7)
    plt.title('Word Overlap Ratio Distribution')
    plt.xlabel('Word Overlap Ratio')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 4)
    plt.hist(tech_term_retention, bins=20, alpha=0.7)
    plt.title('Technical Term Retention Distribution')
    plt.xlabel('Technical Term Retention')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('../dataset/qa_data/summary_evaluation.png')
    
    # Print statistics
    print("\n--- Summary Evaluation Statistics ---\n")
    
    print("Compression Ratio (summary length / original length):")
    print(f"  Mean: {stats['compression_ratio']['mean']:.2f} (Target: 0.3-0.6)")
    print(f"  Std Dev: {stats['compression_ratio']['std']:.2f}")
    print(f"  Min: {stats['compression_ratio']['min']:.2f}")
    print(f"  Max: {stats['compression_ratio']['max']:.2f}")
    
    print("\nCosine Similarity (semantic similarity):")
    print(f"  Mean: {stats['similarity_score']['mean']:.2f} (Higher is better)")
    print(f"  Std Dev: {stats['similarity_score']['std']:.2f}")
    print(f"  Min: {stats['similarity_score']['min']:.2f}")
    print(f"  Max: {stats['similarity_score']['max']:.2f}")
    
    print("\nWord Overlap Ratio (% of original words retained):")
    print(f"  Mean: {stats['word_overlap']['mean']:.2f} (Higher indicates better information retention)")
    print(f"  Std Dev: {stats['word_overlap']['std']:.2f}")
    print(f"  Min: {stats['word_overlap']['min']:.2f}")
    print(f"  Max: {stats['word_overlap']['max']:.2f}")
    
    print("\nTechnical Term Retention:")
    print(f"  Mean: {stats['tech_term_retention']['mean']:.2f} (Higher indicates better technical content preservation)")
    print(f"  Std Dev: {stats['tech_term_retention']['std']:.2f}")
    print(f"  Min: {stats['tech_term_retention']['min']:.2f}")
    print(f"  Max: {stats['tech_term_retention']['max']:.2f}")
    
    print("\n--- Sample Original vs Summary Comparisons ---\n")
    for i, sample in enumerate(samples):
        print(f"Sample {i+1} - Task: {sample['task']}")
        print(f"  Compression: {sample['comp_ratio']:.2f}, Similarity: {sample['similarity']:.2f}")
        print(f"  Original: {sample['original'][:200]}...")
        print(f"  Summary: {sample['summary']}")
        print()
    
    # Save results to CSV for further analysis
    results = []
    for i, item in enumerate(data):
        if i < len(compression_ratios):  # Ensure indices match
            results.append({
                'Task': item.get('task', 'Unknown'),
                'Compression_Ratio': compression_ratios[i],
                'Similarity_Score': similarity_scores[i],
                'Word_Overlap': word_overlap_ratios[i],
                'Tech_Term_Retention': tech_term_retention[i]
            })
    
    pd.DataFrame(results).to_csv('../dataset/qa_data/summary_evaluation.csv', index=False)
    print(f"Detailed evaluation saved to '../dataset/qa_data/summary_evaluation.csv'")
    print(f"Plots saved to '../dataset/qa_data/summary_evaluation.png'")

if __name__ == "__main__":
    evaluate_summaries()
