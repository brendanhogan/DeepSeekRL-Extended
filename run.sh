#!/bin/bash

# Example 1: Run baseline evaluation for GSM8K with GPT-4.1-nano
# python baseline_eval.py --dataset "gsm8k" --num_samples 50

# Example 2: Run baseline evaluation for MATH-500 with GPT-4.1-nano
# python baseline_eval.py --dataset "math500" --num_samples 50

# Example 2b: Run baseline evaluation for MATH-500 with local Qwen2.5-7B-Instruct
# python baseline_eval.py --dataset "math500" --use_local_inference_model --num_samples 50

# Example 3: Train strategy advisor on GSM8K
# python main.py --output_dir "gsm8k_strategy_training" --dataset_name "gsm8k" --evaluator "gsm8k" --verbose

# Example 4: Train strategy advisor on MATH-500 with GPT-4.1-nano
# python main.py --output_dir "math500_strategy_training" --dataset_name "math500" --evaluator "math500" --verbose

# Example 5: Train strategy advisor on MATH-500 with local Qwen2.5-7B-Instruct  
# (Uses simplified prompting: system="You are Qwen, a helpful assistant", all instructions in user prompt)
# python main.py --output_dir "math500_local_training" --dataset_name "math500" --evaluator "math500" --use_local_inference_model --verbose
# python main.py --output_dir "math500_local_training_2" --dataset_name "math500" --evaluator "math500" --use_local_inference_model --verbose

# python plotter.py --log_dir "math500_local_training"

python plotter.py --log_dir "math500_local_training_2"