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
# python main.py --output_dir "svg_testing_8" --verbose --dataset_name "svg" --evaluator "svg"
# python main.py --output_dir "final_real_full_run_14b" --verbose --dataset_name "svg" --evaluator "svg"
# python main.py --output_dir "less_pdf_runs" --verbose --dataset_name "svg" --evaluator "svg"
# python main.py --output_dir "final_qwen_3_correct_prompt_final" --verbose --dataset_name "svg" --evaluator "svg"
python plotter.py --log_dir "final_qwen_3_correct_prompt_final.5-7B"
