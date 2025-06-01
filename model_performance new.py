import json
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import os
import time
from openai import OpenAI
import re
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

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

def agent_communication(client, agent1, agent2, question, ground_truth):
    """Enable any two agents to communicate and collaborate on solving a problem."""
    try:
        total_time = 0
        
        # First agent's initial response
        start_time = time.time()
        if "claude" in agent1.lower():
            response1 = client.chat.completions.create(
                model=agent1,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that solves math word problems step by step."},
                    {"role": "user", "content": question}
                ],
                temperature=0.7
            )
        else:
            response1 = client.chat.completions.create(
                model=agent1,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that solves math word problems step by step."},
                    {"role": "user", "content": question}
                ]
            )
        end_time = time.time()
        total_time += (end_time - start_time)
        answer1 = response1.choices[0].message.content
        
        # Second agent reviews first agent's response
        start_time = time.time()
        if "claude" in agent2.lower():
            response2 = client.chat.completions.create(
                model=agent2,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that reviews and improves math problem solutions."},
                    {"role": "user", "content": f"Here's a math problem and another agent's solution. Please review it and provide your thoughts:\n\nProblem: {question}\n\nFirst agent's solution: {answer1}"}
                ],
                temperature=0.7
            )
        else:
            response2 = client.chat.completions.create(
                model=agent2,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that reviews and improves math problem solutions."},
                    {"role": "user", "content": f"Here's a math problem and another agent's solution. Please review it and provide your thoughts:\n\nProblem: {question}\n\nFirst agent's solution: {answer1}"}
                ]
            )
        end_time = time.time()
        total_time += (end_time - start_time)
        review2 = response2.choices[0].message.content
        
        # First agent considers the review
        start_time = time.time()
        if "claude" in agent1.lower():
            response3 = client.chat.completions.create(
                model=agent1,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that solves math word problems step by step."},
                    {"role": "user", "content": f"Here's a math problem and a review of your solution. Please provide an improved solution:\n\nProblem: {question}\n\nYour original solution: {answer1}\n\nReview: {review2}"}
                ],
                temperature=0.7
            )
        else:
            response3 = client.chat.completions.create(
                model=agent1,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that solves math word problems step by step."},
                    {"role": "user", "content": f"Here's a math problem and a review of your solution. Please provide an improved solution:\n\nProblem: {question}\n\nYour original solution: {answer1}\n\nReview: {review2}"}
                ]
            )
        end_time = time.time()
        total_time += (end_time - start_time)
        final_answer = response3.choices[0].message.content
        
        # Extract the final numerical answer
        model_answer = extract_final_answer(final_answer)
        ground_truth_answer = extract_final_answer(ground_truth)
        is_correct = model_answer == ground_truth_answer
        
        return {
            'question': question,
            'ground_truth': ground_truth,
            'ground_truth_answer': ground_truth_answer,
            'agent1_initial_response': answer1,
            'agent2_review': review2,
            'final_response': final_answer,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'response_time': total_time,  # Total time for all three steps
            'agents_used': f"{agent1} + {agent2}"
        }
    except Exception as e:
        print(f"Error in agent communication: {e}")
        return None

def select_agent_pairs(models):
    """Let user select which agent pairs to test."""
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    pairs = []
    while True:
        print("\nSelect two models to pair (enter numbers, or 'done' to finish):")
        try:
            choice = input("Enter two numbers separated by space (e.g., '1 2') or 'done': ")
            if choice.lower() == 'done':
                break
            
            num1, num2 = map(int, choice.split())
            if 1 <= num1 <= len(models) and 1 <= num2 <= len(models):
                agent1 = models[num1-1]
                agent2 = models[num2-1]
                pairs.append((agent1, agent2))
                print(f"Added pair: {agent1} + {agent2}")
            else:
                print("Invalid model numbers. Please try again.")
        except ValueError:
            print("Invalid input. Please enter two numbers separated by space.")
    
    return pairs

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

def evaluate_step_by_step_accuracy(response, ground_truth):
    """Evaluate the accuracy of the step-by-step solution process."""
    try:
        # Extract steps from the response
        steps = [step.strip() for step in response.split('\n') if step.strip()]
        
        # Extract numbers from each step
        step_numbers = []
        for step in steps:
            numbers = re.findall(r'\d+\.?\d*', step)
            if numbers:
                step_numbers.append([float(num) for num in numbers])
        
        # Extract numbers from ground truth
        ground_truth_numbers = re.findall(r'\d+\.?\d*', ground_truth)
        ground_truth_numbers = [float(num) for num in ground_truth_numbers]
        
        # Calculate step accuracy
        correct_steps = 0
        total_steps = len(step_numbers)
        
        for step_nums in step_numbers:
            # Check if any number in the step matches a number in ground truth
            if any(num in ground_truth_numbers for num in step_nums):
                correct_steps += 1
        
        return (correct_steps / total_steps * 100) if total_steps > 0 else 0
    except Exception as e:
        print(f"Error in step-by-step evaluation: {e}")
        return 0

def calculate_metrics(results):
    """Calculate performance metrics from results."""
    if not results:
        return {
            'total_problems': 0,
            'correct_answers': 0,
            'accuracy': 0,
            'avg_response_time': 0,
            'step_accuracy': 0
        }
    
    total_problems = len(results['results'])
    correct_answers = sum(1 for r in results['results'] if r.get('is_correct'))
    
    # Calculate average response time
    response_times = [r.get('response_time', 0) for r in results['results']]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    # Calculate step-by-step accuracy, handling both individual and combined models
    step_accuracies = []
    for r in results['results']:
        # For combined models, use final_response instead of model_response
        response_text = r.get('final_response', r.get('model_response'))
        if response_text and 'ground_truth' in r:
            step_accuracy = evaluate_step_by_step_accuracy(response_text, r['ground_truth'])
            step_accuracies.append(step_accuracy)
    avg_step_accuracy = sum(step_accuracies) / len(step_accuracies) if step_accuracies else 0
    
    return {
        'total_problems': total_problems,
        'correct_answers': correct_answers,
        'accuracy': (correct_answers / total_problems * 100) if total_problems > 0 else 0,
        'avg_response_time': avg_response_time,
        'step_accuracy': avg_step_accuracy
    }

def normalize(data):
    if max(data) == min(data):
        return [0 for _ in data]  # Return zeros if all values are the same
    return [(x - min(data)) / (max(data) - min(data)) * 100 for x in data]

def visualize_performance(results):
    """Create a dashboard PNG with all model performance charts using Matplotlib."""
    if not results:
        print("No results to visualize!")
        return

    # Prepare data
    models = []
    accuracies = []
    step_accuracies = []
    response_times = []
    for model, data in results.items():
        metrics = calculate_metrics(data)
        models.append(model)
        accuracies.append(metrics['accuracy'])
        step_accuracies.append(metrics['step_accuracy'])
        response_times.append(metrics['avg_response_time'])

    # Set up the 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=20)

    # Accuracy Bar Chart
    axs[0, 0].bar(models, accuracies, color='skyblue')
    axs[0, 0].set_title('Model Accuracy Comparison')
    axs[0, 0].set_ylabel('Accuracy (%)')
    axs[0, 0].set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    for i, v in enumerate(accuracies):
        axs[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    # Step Accuracy Bar Chart
    axs[0, 1].bar(models, step_accuracies, color='orange')
    axs[0, 1].set_title('Model Step Accuracy Comparison')
    axs[0, 1].set_ylabel('Step Accuracy (%)')
    axs[0, 1].set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    for i, v in enumerate(step_accuracies):
        axs[0, 1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    # Response Time Bar Chart
    axs[1, 0].bar(models, response_times, color='salmon')
    axs[1, 0].set_title('Model Response Time Comparison')
    axs[1, 0].set_ylabel('Avg Response Time (seconds)')
    axs[1, 0].set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    for i, v in enumerate(response_times):
        axs[1, 0].text(i, v + 0.1, f'{v:.2f}s', ha='center', va='bottom', fontsize=9)

    # Radar Chart for Overall Performance
    # Normalize metrics for radar chart (0-100 scale)
    def norm(arr):
        arr = np.array(arr)
        if arr.max() == arr.min():
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min()) * 100
    acc_norm = norm(accuracies)
    step_acc_norm = norm(step_accuracies)
    time_norm = 100 - norm(response_times)  # Lower time is better
    radar_labels = ['Accuracy', 'Step Accuracy', 'Speed']
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]
    axs[1, 1] = plt.subplot(224, polar=True)
    for i, model in enumerate(models):
        values = [acc_norm[i], step_acc_norm[i], time_norm[i]]
        values += values[:1]
        axs[1, 1].plot(angles, values, label=model)
        axs[1, 1].fill(angles, values, alpha=0.1)
    axs[1, 1].set_title('Overall Performance Comparison', y=1.1)
    axs[1, 1].set_xticks(angles[:-1])
    axs[1, 1].set_xticklabels(radar_labels)
    axs[1, 1].set_yticklabels([])
    axs[1, 1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('model_performance_comparison.png', dpi=200)
    plt.close()
    print("Saved: model_performance_comparison.png")

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
            f"{metrics['step_accuracy']:.2f}%",
            f"{metrics['avg_response_time']:.2f}s"
        ])
    
    # Sort by accuracy
    table_data.sort(key=lambda x: float(x[2].strip('%')), reverse=True)
    
    # Print table
    print("\nComprehensive Model Performance Comparison")
    print("=" * 100)
    print(tabulate(table_data,
                  headers=['Model', 'Problems Solved', 'Accuracy', 'Step Accuracy', 'Avg Response Time'],
                  tablefmt='grid'))
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_results_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

def main():
    print("Comprehensive Model Performance Analysis")
    print("=" * 80)
    
    # Initialize client
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
    test_problems = test_df.sample(n=num_problems)
    
    # Store results for each model
    all_results = {}
    
    print(f"\nTesting {len(models)} models with {len(test_problems)} problems each...")
    
    # Test individual models
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
    
    # Test agent communication
    print("\nWould you like to test agent communication? (yes/no)")
    if input().lower() == 'yes':
        print("\nSelect which agent pairs to test:")
        agent_pairs = select_agent_pairs(models)
        
        for agent1, agent2 in agent_pairs:
            print(f"\nTesting communication between {agent1} and {agent2}...")
            communication_results = []
            
            for idx, row in test_problems.iterrows():
                result = agent_communication(client, agent1, agent2, row['question'], row['answer'])
                if result:
                    communication_results.append(result)
                    print(f"Processed problem with {agent1} and {agent2}")
            
            if communication_results:
                all_results[f"{agent1}+{agent2}"] = {
                    'results': communication_results,
                    'accuracy': sum(1 for r in communication_results if r['is_correct']) / len(communication_results) * 100
                }
    
    # Display results
    if all_results:
        display_comprehensive_table(all_results)
        visualize_performance(all_results)
    else:
        print("No results to display!")

if __name__ == "__main__":
    main() 