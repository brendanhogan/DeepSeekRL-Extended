# python main.py --output_dir "final1" --verbose
# python main.py --output_dir "debate_gpt4o_mini_final_run_2" --verbose --resume
# python plotter.py --log_dir "debate_gpt4o_mini_final_run_2"
# python separate_judge_eval.py --output_dir "debate_gpt4o_mini_final_run_2" --verbose
# python main.py --output_dir "ld_gpt4o_mini_gpt_judge" --verbose --dataset_name "LD" --evaluator "LD" --judge_model_name gpt-4o-mini --resume
# python main.py --output_dir "ld_gpt4o_mini_gpt_judge_llama_8b" --verbose --dataset_name "LD" --evaluator "LD" --judge_model_name gpt-4o-mini --model_name Qwen/Qwen2.5-7B-Instruct
python plotter.py --log_dir "ld_gpt4o_mini_gpt_judge_llama_8b"
