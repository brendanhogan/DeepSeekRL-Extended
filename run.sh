# python main.py --output_dir "final1" --verbose
# python main.py --output_dir "debate_gpt4o_mini_final_run_2" --verbose --resume
# python plotter.py --log_dir "debate_gpt4o_mini_final_run_2"
# python separate_judge_eval.py --output_dir "debate_gpt4o_mini_final_run_2" --verbose
# python main.py --output_dir "ld_gpt4o_mini_gpt_judge" --verbose --dataset_name "LD" --evaluator "LD" --judge_model_name gpt-4o-mini --resume
# python main.py --output_dir "ld_gpt4o_mini_gpt_judge_llama_8b" --verbose --dataset_name "LD" --evaluator "LD" --judge_model_name gpt-4o-mini --model_name Qwen/Qwen2.5-7B-Instruct
# python main.py --output_dir "chopped_gpt4o_mini_gpt_judge_Qwen2.5-7B" --verbose --dataset_name "chopped" --evaluator "chopped" --judge_model_name gpt-4o-mini --model_name Qwen/Qwen2.5-7B-Instruct --resume
# python main.py --output_dir "chopped_gpt4o_mini_gpt_judge_Qwen2.5-1B" --verbose --dataset_name "chopped" --evaluator "chopped"
# python plotter.py --log_dir "ld_gpt4o_mini_gpt_judge_llama_8b"
# python plotter.py --log_dir "chopped_gpt4o_mini_gpt_judge_Qwen2.5-7B"
# python main.py --output_dir "grm_test" --verbose
python plotter.py --log_dir "grm_test"
