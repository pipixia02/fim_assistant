from retrieval.hybrid_retrieval import HybridRetriever
from create_faissdb import load_vector_store
import argparse
import os
from loguru import logger
from openai import OpenAI
from qwen_chat import call_qwen_api, api_retry
import json
import time
from tqdm import tqdm

model_name = 'Qwen2.5-7B-Instruct-local'

# Configure logging
logger.remove()  # Remove default console output
logger.add(sink=lambda msg: print(msg), level="INFO")  # Add console output

class RAGSystem:
    def __init__(self, model_name="Qwen2.5-7B-Instruct-local", 
                 retrieval_model_path="Models/retrieval_models/prompt1_model/model_0.067",
                 faiss_index_path="database/faiss_prompt1",
                 use_custom_model=True,
                 use_reranker=False,
                 use_intersection=True):
        """
        Initialize RAG system
        
        Args:
            model_name: LLM model name
            retrieval_model_path: Retrieval model path
            faiss_index_path: FAISS index path
            use_custom_model: Whether to use custom retrieval model
            use_reranker: Whether to use reranker
            use_intersection: Whether to use intersection retrieval
        """
        self.model_name = model_name
        self.use_intersection = use_intersection
        
        # Load vector store
        logger.info(f"Loading vector store from {faiss_index_path}...")
        try:
            self.vectordb = load_vector_store(
                model_path=retrieval_model_path,
                faiss_index_path=faiss_index_path,
                use_custom_model=use_custom_model
            )
            logger.info(f"Vector store loaded successfully using model: {retrieval_model_path}")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
            
        # Initialize retriever
        self.retriever = HybridRetriever(self.vectordb, use_reranker=use_reranker)
        logger.info(f"Hybrid retriever initialized with intersection retrieval {'enabled' if use_intersection else 'disabled'}")
        
        # Initialize OpenAI client (for local model)
        try:
            self.client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="sk-xxx",  # Placeholder, doesn't affect local calls
            )
            logger.info(f"LLM client initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    def retrieve(self, query, top_k=3):
        """
        Retrieve relevant documents
        
        Args:
            query: User query
            top_k: Number of documents to return
            
        Returns:
            Retrieved documents list
        """
        try:
            logger.info(f"Retrieving with {'intersection' if self.use_intersection else 'standard hybrid'} algorithm: '{query}'")
            
            # Use search method with intersection parameter
            results = self.retriever.search(
                query=query, 
                top_k=top_k, 
                use_intersection=self.use_intersection,
                return_format="list"
            )
            
            if not results:
                logger.warning("No relevant documents found")
                return [], ""
            
            # Calculate total characters
            total_chars = sum(len(result.get('content', '')) for result in results)
            logger.info(f"Retrieved {len(results)} documents, total characters: {total_chars}")
            
            # Truncate if content is too long
            max_chars = 6000
            if total_chars > max_chars:
                logger.warning(f"Content too long ({total_chars} chars), truncating")
                truncated_results = []
                current_chars = 0
                
                # Prioritize documents with higher scores
                sorted_results = sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)
                
                for result in sorted_results:
                    content_len = len(result.get('content', ''))
                    # If a single document exceeds half the threshold
                    if content_len > max_chars / 2:
                        # Keep beginning and end of document
                        truncated_content = result['content'][:int(max_chars/4)] + "\n...[Content truncated]...\n" + result['content'][-int(max_chars/4):]
                        result['content'] = truncated_content
                        result['truncated'] = True
                        current_chars += len(truncated_content)
                        truncated_results.append(result)
                    # If adding this document doesn't exceed the threshold
                    elif current_chars + content_len <= max_chars:
                        current_chars += content_len
                        truncated_results.append(result)
                    # Otherwise skip this document
                    else:
                        continue
                        
                results = truncated_results
                logger.info(f"After truncation: {len(results)} documents, {current_chars} characters")
            
            # Format output
            retrieval_text = ""
            for i, result in enumerate(results):
                retrieval_text += f"\n### Related documents {i+1}:\n"
                
                # Add truncation notice
                if result.get('truncated', False):
                    retrieval_text += "[Note: This document has been truncated]\n"
                    
                retrieval_text += f"Content: {result['content']}\n"
                if 'original_procedure' in result['metadata']:
                    # Truncate procedure if too long
                    procedure = result['metadata']['original_procedure']
                    if len(procedure) > 1000:
                        procedure = procedure[:500] + "\n...[Procedure content truncated]...\n" + procedure[-500:]
                    retrieval_text += f"Fault Isolation Procedure: {procedure}\n"
                if 'original_initial_evaluation' in result['metadata']:
                    # Truncate evaluation if too long
                    evaluation = result['metadata']['original_initial_evaluation']
                    if len(evaluation) > 1000:
                        evaluation = evaluation[:500] + "\n...[Evaluation content truncated]...\n" + evaluation[-500:]
                    retrieval_text += f"Initial Evaluation: {evaluation}\n"
                
                # Add source and score info
                source = result.get('source', 'unknown')
                score = result.get('final_score', 0)
                retrieval_text += f"Source: {source}, Score: {score:.4f}\n"
            
            return results, retrieval_text
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return [], ""

    def format_prompt(self, query, context):
        """
        Format prompt
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        # Calculate context length
        context_length = len(context)
        logger.info(f"Context length: {context_length} characters")
        
        # Add notice if context is too long
        if context_length > 6000:
            context_note = "\n[Note: Due to content length, some documents have been truncated. Please answer based on available information.]\n"
            context = context_note + context
        
        prompt = f"""You are a professional aviation maintenance consultant. 
        Your answers should be accurate, professional, and in compliance with aviation industry standards. 
        Please answer the following questions regarding aviation maintenance.
        You should base your answers primarily on the relevant documents provided, which contain fault descriptions and maintenance procedures. 
        If the documents are insufficient to answer the question, you may also answer based on your professional knowledge. 
        Please ensure that your answers are accurate, professional, and practical.
        
Related documents:
{context}

User question:
{query}

Please provide a professional and accurate answer:
"""
        return prompt

    def process_query(self, query, top_k=3):
        """
        Process user query
        
        Args:
            query: User query
            top_k: Number of documents to return
            
        Returns:
            Answer
        """
        try:
            # Retrieve relevant documents
            results, context = self.retrieve(query, top_k)
            
            if not results:
                return "Sorry, I couldn't find any relevant information for your question. Please try rephrasing or provide more details."
            
            # Format prompt
            prompt = self.format_prompt(query, context)
            
            # Call LLM
            answer = self.call_llm(prompt)
            
            return answer
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return f"Sorry, an error occurred while processing your question: {e}"

    def call_llm(self, prompt):
        """
        Call LLM to generate answer

        Args:
            prompt: Prompt text

        Returns:
            LLM generated answer
        """
        try:
            logger.info("Calling LLM to generate answer")
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Sorry, an error occurred while generating the answer. Please try again later."

def main():
    parser = argparse.ArgumentParser(description="Aviation Maintenance Q&A System")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct-local",
                        help="LLM model name")
    parser.add_argument("--retrieval_model", type=str, 
                        default="Models/retrieval_models/prompt1_model/model_0.067",
                        help="Retrieval model path")
    parser.add_argument("--faiss_index", type=str, default="database/faiss_prompt1",
                        help="FAISS index path")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of retrieval results")

    parser.add_argument("--use_intersection", action="store_true", default=True,
                        help="Use intersection retrieval")
    parser.add_argument("--batch_mode", action="store_true", default=True,
                        help="Process dataset in batch mode")
    parser.add_argument("--dataset_path", type=str, 
                        default="dataset/retrieval_data/prompt1/qa_test.json",
                        help="Test dataset path")
    parser.add_argument("--output_file", type=str, 
                        default="results/intersection_retrieval_results.json",
                        help="Output file path")
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="Run in interactive mode")

    
    args = parser.parse_args()
    
    # Check if vLLM service is running
    if not os.popen("curl -s http://localhost:8000/v1/models").read():
        logger.warning("Local vLLM service appears to be offline. Please ensure it's running:")
        logger.warning("python -m vllm.entrypoints.openai.api_server --model Models/Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen2.5-7B-Instruct-local")
    
    try:
        # Initialize RAG system
        rag_system = RAGSystem(
            model_name=args.model_name,
            retrieval_model_path=args.retrieval_model,
            faiss_index_path=args.faiss_index,
            use_reranker=False,
            use_intersection=args.use_intersection
        )
        
        if args.batch_mode:
            # Batch processing mode
            process_dataset(rag_system, args)
        elif args.interactive:
            # Interactive mode
            logger.info("Interactive Q&A mode started. Enter 'exit' or 'quit' to end.")
            while True:
                query = input("\nEnter your question: ")
                if query.lower() in ["exit", "quit", "q"]:
                    break
                
                print("\nThinking...")
                answer = rag_system.process_query(query, args.top_k)
                print(f"\nAnswer: {answer}")
        else:
            logger.info("No mode specified, defaulting to interactive mode")
            logger.info("Interactive Q&A mode started. Enter 'exit' or 'quit' to end.")
            while True:
                query = input("\nEnter your question: ")
                if query.lower() in ["exit", "quit", "q"]:
                    break
                
                print("\nThinking...")
                answer = rag_system.process_query(query, args.top_k)
                print(f"\nAnswer: {answer}")
            
    except Exception as e:
        logger.error(f"System error: {e}")

def process_dataset(rag_system, args):
    """
    Process dataset in batch mode and save results
    
    Args:
        rag_system: RAG system instance
        args: Command line arguments
    """
    logger.info(f"Batch mode started, processing dataset: {args.dataset_path}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Load dataset
    try:
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"Dataset loaded with {len(dataset)} questions")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Process each question
    results = []
    for i, item in enumerate(tqdm(dataset, desc="Processing questions")):
        question = item.get('question', '')
        if not question:
            logger.warning(f"Skipping question {i+1}, empty question")
            continue
        
        logger.info(f"Processing question {i+1}/{len(dataset)}: {question[:50]}...")
        
        try:
            # Process question
            start_time = time.time()
            answer = rag_system.process_query(question, args.top_k)
            end_time = time.time()
            
            # Record result
            result = {
                'question': question,
                'answer': answer,
                'ground_truth': item.get('answer', ''),
                'processing_time': end_time - start_time,
                'metadata': {
                    'original_description': item.get('original_description', ''),
                    'original_procedure': item.get('original_procedure', ''),
                    'task': item.get('task', ''),
                    'original_initial_evaluation': item.get('original_initial_evaluation', '')
                }
            }
            results.append(result)
            
            # Save results periodically
            if (i + 1) % 10 == 0 or i == len(dataset) - 1:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved current results to {args.output_file}")
                
            # Add delay to avoid API rate limits
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Failed to process question {i+1}: {e}")
            # Record failed question
            result = {
                'question': question,
                'answer': f"Processing failed: {str(e)}",
                'ground_truth': item.get('answer', ''),
                'error': str(e),
                'metadata': {
                    'original_description': item.get('original_description', ''),
                    'original_procedure': item.get('original_procedure', ''),
                    'task': item.get('task', ''),
                    'original_initial_evaluation': item.get('original_initial_evaluation', '')
                }
            }
            results.append(result)
    
    # Save final results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Batch processing completed, processed {len(results)} questions, results saved to {args.output_file}")

if __name__ == "__main__":
    main()
