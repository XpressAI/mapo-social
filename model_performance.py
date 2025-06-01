import json
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import os
import time
from openai import OpenAI
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
XPRESSAI_API_KEY = os.getenv("XPRESSAI_API_KEY")
XPRESSAI_BASE_URL = "https://relay.public.cloud.xpress.ai/v1"

def get_xpressai_models():
    """Get list of available XpressAI models."""
    # Predefined list of models we know are available
    return [
        "gpt-3.5-turbo",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest"
    ]

def extract_final_answer(text):
    """Extract the final numerical answer from a string."""
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[-1])
    return None

def process_problem_with_model(client, model, question, ground_truth):
    """Process a single problem with a specific model."""
    try:
        start_time = time.time()
        
        # Add temperature parameter for Claude models
        if "claude" in model.lower():
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that solves math word problems step by step."},
                    {"role": "user", "content": question}
                ],
                temperature=0.7  # Add temperature parameter for Claude models
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that solves math word problems step by step."},
                    {"role": "user", "content": question}
                ]
            )
        
        end_time = time.time()
        
        # Print raw response for debugging
        print(f"Raw response for {model}: {response}")
        
        # Extract the answer from the response
        model_response = response.choices[0].message.content
        model_answer = extract_final_answer(model_response)
        ground_truth_answer = extract_final_answer(ground_truth)
        is_correct = model_answer == ground_truth_answer
        
        return {
            'question': question,
            'ground_truth': ground_truth,
            'ground_truth_answer': ground_truth_answer,
            'model_response': model_response,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'response_time': end_time - start_time,
            'model_used': model
        }
    except Exception as e:
        print(f"Error processing problem with model {model}: {e}")
        return None

def test_all_models():
    """Test all available XpressAI models with sample problems."""
    # Initialize client with hardcoded API key and base URL
    client = OpenAI(
        base_url=XPRESSAI_BASE_URL,
        api_key=XPRESSAI_API_KEY
    )
    
    # Get available models
    models = get_xpressai_models()
    if not models:
        print("No models available!")
        return
    
    # Load sample problems from GSM8K
    from datasets import load_dataset
    dataset = load_dataset("gsm8k", "main")
    test_df = pd.DataFrame(dataset['test'])
    
    # Ask user for the number of problems
    num_problems = int(input("How many problems do you want each model to solve? "))
    test_problems = test_df.sample(n=num_problems)  # Use user-specified number of problems
    
    # Store results for each model
    all_results = {}
    
    print(f"\nTesting {len(models)} models with {len(test_problems)} problems each...")
    
    for model in models:
        print(f"\nTesting model: {model}")
        model_results = []
        
        for idx, row in test_problems.iterrows():
            result = process_problem_with_model(client, model, row['question'], row['answer'])
            if result:
                model_results.append(result)
                print(f"Processed problem with {model}")
        
        if model_results:
            all_results[model] = {
                'results': model_results,
                'accuracy': sum(1 for r in model_results if r['is_correct']) / len(model_results) * 100
            }
    
    return all_results

def display_comprehensive_table(results):
    """Display comprehensive performance table for all models."""
    if not results:
        print("No results to display!")
        return
    
    # Prepare table data
    table_data = []
    for model, data in results.items():
        metrics = calculate_metrics(data)
        table_data.append([
            model,
            metrics['total_problems'],
            f"{metrics['accuracy']:.2f}%",
            f"{metrics['avg_response_time']:.2f}s",
            f"{metrics['success_rate']:.2f}%"
        ])
    
    # Sort by accuracy
    table_data.sort(key=lambda x: float(x[2].strip('%')), reverse=True)
    
    # Print table
    print("\nComprehensive Model Performance Comparison")
    print("=" * 100)
    print(tabulate(table_data,
                  headers=['Model', 'Problems Solved', 'Accuracy', 'Avg Response Time', 'Success Rate'],
                  tablefmt='grid'))
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_results_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

def calculate_metrics(results):
    """Calculate performance metrics from results."""
    if not results:
        return {
            'total_problems': 0,
            'correct_answers': 0,
            'accuracy': 0,
            'avg_response_time': 0,
            'success_rate': 0
        }
    
    total_problems = len(results['results'])
    correct_answers = sum(1 for r in results['results'] if r['is_correct'])
    
    # Calculate average response time
    response_times = [r.get('response_time', 0) for r in results['results']]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    return {
        'total_problems': total_problems,
        'correct_answers': correct_answers,
        'accuracy': (correct_answers / total_problems * 100) if total_problems > 0 else 0,
        'avg_response_time': avg_response_time,
        'success_rate': (correct_answers / total_problems * 100) if total_problems > 0 else 0
    }

def main():
    print("Comprehensive Model Performance Analysis")
    print("=" * 80)
    
    # Test all models
    results = test_all_models()
    
    # Display results
    if results:
        display_comprehensive_table(results)
    else:
        print("No results to display!")

if __name__ == "__main__":
    main() 