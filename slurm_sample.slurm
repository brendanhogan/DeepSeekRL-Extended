#!/bin/bash
#SBATCH --job-name=new_generation_method_loss_on_all_tokens
#SBATCH --gres=gpu:h100:1     # Request 1× H100 GPU
#SBATCH --time=24:00:00               # HH:MM:SS
#SBATCH --array=0-2  # 5 tasks × 3 true_val combinations x 2 num_chains × 1 random seeds = 30 experiments
#SBATCH --output=std/%x_%A_%a.out   # Stdout/stderr per-array-task

# Optional: load modules / activate environment
source venv/bin/activate
cd GRPO_based_soft_thinking

K_vals=(2) 
tasks=( "acre" "graph_color" "family_relationships" "color_cube_rotation" "zebra_puzzles" "number_sequence"  "countdown" "prime_factorization" "circuit_logic" "syllogism" "self_reference" "simple_integration"     )
# Calculate indices for 3-dimensional iteration: task × true_vals × randomseed
randomseed_index=0 #$((SLURM_ARRAY_TASK_ID % 1))
chain_index=1 #$((SLURM_ARRAY_TASK_ID % 15))
true_val_index=$(((SLURM_ARRAY_TASK_ID % 3)))
task_index=$((SLURM_ARRAY_TASK_ID / 3))

task=${tasks[$task_index]}
dataset="${task}.reasoning_gym"
K_index=0 # $((SLURM_ARRAY_TASK_ID / 6))
K=${K_vals[$K_index]}
num_chains_vals=(5 10)
num_chains=10 # ${num_chains_vals[$chain_index]}
vanilla_true_vals=(0 0 1)
dirichlet_true_vals=(1 0 0)

num_steps=1000
max_length=768
eval_iterations=50
max_prompt_length=1024
max_completion_length=1024
temperature=0.9
randomseed_vals=(420394 69420)
randomseed=${randomseed_vals[$randomseed_index]}
vanilla_true=${vanilla_true_vals[$true_val_index]}
dirichlet_true=${dirichlet_true_vals[$true_val_index]}

if [ $vanilla_true -eq 1 ]; then
    experiment_name="${dataset}_1.5b_vanilla_${num_steps}_steps_eval_${eval_iterations}_eval_rg_word_count_num_chains_${num_chains}_randomseed_${randomseed}_temperature_${temperature}"
elif [ $dirichlet_true -eq 1 ]; then
    experiment_name="${dataset}_1.5b_dirichlet_${num_steps}_steps_eval_${eval_iterations}_eval_rg_word_count_num_chains_${num_chains}_randomseed_${randomseed}_temperature_${temperature}"
else
    experiment_name="${dataset}_1.5b_different_tokens_${num_steps}_steps_eval_${eval_iterations}_eval_rg_word_count_num_chains_${num_chains}_randomseed_${randomseed}_temperature_${temperature}"
fi

learning_rate=5e-6
model_name="Qwen/Qwen2.5-1.5B-Instruct"
echo "[$(date)] Starting task $SLURM_ARRAY_TASK_ID with T_e=$T_E, k=$K on GPU $CUDA_VISIBLE_DEVICES and experiment_name=$experiment_name and model_name=$model_name and num_chains=$num_chains and randomseed=$randomseed"
#experiment_name=${experiment_names[$SLURM_ARRAY_TASK_ID]}
python main.py \
  --output_dir output_sgd/mixture_grpo_${experiment_name}_${K}\
  --model_name $model_name \
  --mixture_k $K \
  --num_chains $num_chains \
  --num_train_iters $num_steps \
  --learning_rate $learning_rate \
  --experiment_name $experiment_name \
  --normal_generation $vanilla_true \
  --dataset $dataset \
  --eval_iterations $eval_iterations \
  --temperature $temperature \
  --max_prompt_length $max_prompt_length \
  --max_completion_length $max_completion_length \
  --seed $randomseed

echo "[$(date)] Finished task $SLURM_ARRAY_TASK_ID" 