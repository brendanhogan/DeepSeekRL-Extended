# Strategy-Based Training System

This is an implementation of a novel training approach where we train a "strategy advisor" model to give advice to a fixed "solver" model (GPT-4.1-nano). The advisor gets rewarded based on how well the solver performs with its advice.

## Core Concept

**Traditional approach**: Train model to solve problems directly
**Our approach**: Train model to give strategic advice to another model

## System Architecture

```
Training Model → Strategy Advice → Inference Model → Solution → Reward → Backprop to Training Model
```

### Key Components

1. **Training Model** (Qwen 1.5B): Generates strategic advice
2. **Inference Model** (GPT-4.1-nano): Uses advice to solve problems  
3. **Single Generation**: 1 strategy → 1 solution attempt → success if correct
4. **Gradient Flow**: Only strategy tokens get gradients, not problem understanding

## Pipeline Flow

### 1. Baseline Evaluation (`baseline_eval.py`)
- Tests inference model with default "Think step by step" strategy
- Establishes performance ceiling/baseline
- Saves results to `baseline_evaluation.json` and `.txt`

### 2. Strategy Generation
**Training model prompt:**
```
You give strategic advice to help other models solve math problems. 
Generate 2-3 sentences of helpful strategy.

How should I approach this math problem: [PROBLEM]
```

**Output:** Strategic advice (2-3 sentences)

### 3. Strategy Evaluation  
**Inference model prompt:**
```
You will be given a math problem. Here is some strategic advice:
<strategy>[GENERATED_STRATEGY]</strategy>

Now solve this problem. Format your response as:
<reason>Your reasoning here</reason>
<answer>Your final answer</answer>
```

**Process:**
- Generate 1 solution using the strategy
- Success = solution is correct
- Reward = 2.0 if correct, 0.0 otherwise

### 4. GRPO Loss Computation
- Compute advantages across strategy generations
- Apply GRPO loss **only to strategy tokens**
- KL penalty to prevent drift from reference model

## File Structure

```
├── baseline_eval.py           # Baseline evaluation script  
├── main.py                    # Core training loop (modified)
├── test_strategy_pipeline.py  # Pipeline testing
├── run.sh                     # Complete workflow
├── llms.py                    # Inference interface (enhanced)
├── plotter.py                 # Visualization (updated)
└── STRATEGY_README.md         # This file
```

## Key Modifications

### `main.py` - New Functions:
- `generate_strategies()`: Generate strategic advice from training model
- `evaluate_strategy_with_inference()`: Single generation evaluation with inference model
- `score_strategies()`: Compute rewards based on inference model performance
- `compute_strategy_loss()`: GRPO loss only over strategy tokens
- `strategy_grpo_loss()`: Complete strategy training pipeline
- `eval_on_test_set()`: Updated for strategy evaluation

### `llms.py` - Enhanced:
- `InferenceModelInterface`: Unified interface for local + OpenAI models
- Automatic retry logic with exponential backoff
- Deterministic generation (temperature=0, seed=42)
- Proper OpenAI API formatting

### `plotter.py` - Updated Metrics:
- `success_rate`: Success rate of strategies
- `mean_reward`: Average strategy rewards
- `strategy_length`: Average strategy length
- `kl`: KL divergence during training

## Usage

### Quick Test
```bash
python test_strategy_pipeline.py
```

### Full Pipeline
```bash
bash run.sh
```

### Individual Steps
```bash
# 1. Baseline evaluation
python baseline_eval.py --inference_model "gpt-4.1-nano" --num_samples 50

# 2. Strategy training  
python main.py --output_dir "strategy_training_v1" --verbose \
               --num_train_iters 200 --eval_iterations 10

# 3. Generate plots
python plotter.py --log_dir "strategy_training_v1"
```

## Training Arguments

Key parameters for strategy training:

```bash
--inference_model "gpt-4.1-nano"     # Fixed solver model
--num_chains 16                      # Strategies per question
--eval_iterations 10                 # Evaluate every N steps
--kl_weight_beta 0.04                # KL penalty weight
--temperature 0.9                    # Strategy generation temp
```

## Expected Outputs

### Training Logs:
- `{round}_strategy_evaluation.json`: Detailed strategy→solution→reward data
- `{round}_strategy_evaluation.txt`: Human-readable strategy evaluations
- `train_logs.json`: Training metrics (pass@2 rate, rewards, etc.)

### Evaluation Logs:
- `strategy_eval_{round}.json`: Test set strategy evaluation results
- `strategy_eval_{round}.txt`: Human-readable evaluation summaries

### Plots:
- Success rate over training
- Mean reward trends  
- Strategy length evolution
- KL divergence tracking
- Learning rate schedule

## Key Insights

1. **Indirect Optimization**: We optimize strategy generation, not direct problem solving
2. **Collaborative Intelligence**: Training model teaches, inference model solves
3. **Token-Level Precision**: Gradients only flow through strategy tokens
4. **Simple Evaluation**: Direct assessment of strategy effectiveness
5. **Clear Separation**: Strategy advice vs problem solving are distinct skills

## Potential Applications

- **Domain Transfer**: Train strategy advisors for new problem types
- **Meta-Learning**: Learn general problem-solving strategies
- **Human-AI Collaboration**: Generate advice for human problem solvers
- **Curriculum Learning**: Progressive strategy complexity
- **Multi-Agent Systems**: Multiple advisors with different specializations

---

This system represents a novel approach to training where we explicitly separate strategic thinking from execution, potentially leading to more robust and transferable problem-solving capabilities. 