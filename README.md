
# DeepSeek R1 Implementation

## Motivation
I wanted to recreate DeepSeek R1's results at a smaller scale, focusing on understanding the core mechanics by implementing everything from scratch. This repository trains Qwen1.5B on various reasoning datasets including the [grade school math dataset](https://github.com/openai/grade-school-math), with extensive enhancements for advanced token generation strategies and multi-dataset support.

This implementation heavily borrows from [Will Brown's  work](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) ([@willccbb](https://x.com/willccbb)), but restructures the code into a format optimized for learning and experimentation.

The key difference in my implementation is computing the GRPO loss function directly rather than using external RL libraries, and reformatting into a multi script repo.

The implementation now includes several major enhancements:
- **Advanced Token Generation**: Mixture-of-token generation with phase transitions, supporting various sampling strategies (nucleus, Dirichlet, element-wise max)
- **Multi-Dataset Support**: Extended beyond GSM8K to include Reasoning Gym datasets, MBPP, LeetCode, and Math500
- **Safe Code Execution**: Subprocess-based code execution for programming tasks with crash protection
- **Enhanced Evaluation**: LLM-based answer evaluation and sophisticated reward functions
- **Comprehensive Logging**: Detailed generation logs with phase transition tracking and token selection analysis

I hope this might help other people understand things better, and maybe provide an easier way to try out smaller scale ideas etc.

## Installation
```
pip install -r requirements.txt
```

Required environment variables:
```
export HUGGINGFACE_TOKEN="your-token-here"
huggingface-cli login
```

## Implementation Details

The system consists of several key modules:

### main.py
Contains the core training loop implementing GRPO (Generalized Reward-Powered Optimization). Features advanced generation capabilities including:
- Mixture-of-token generation with phase transitions
- Multiple sampling strategies (nucleus, Dirichlet, element-wise max)
- Comprehensive logging of generation steps and token selection
- Enhanced loss computation with embedding-based token selection

### llms.py
Manages model loading and configuration with enhanced features:
- Support for LLaMA and Qwen models through Hugging Face's transformers library
- Conditional flash attention support based on model type
- Optimized model loading for different architectures

### rldatasets.py
Handles dataset loading and preprocessing with expanded support:
- GSM8K, Math500, MBPP, and LeetCode datasets
- Full Reasoning Gym integration with task-specific configurations
- Flexible data loaders with custom preprocessing for different data formats

### evaluator.py
Contains evaluation metrics and reward functions with major enhancements:
- LLM-based answer evaluation using OpenAI API
- Safe subprocess execution for programming tasks with crash protection
- Reasoning Gym dataset scoring with custom reward functions
- Timeout handling and robust error management

### utils.py
Utility functions supporting advanced token processing:
- Memory-efficient selective log softmax operations
- Multiple token embedding and log probability computation methods
- Enhanced generation logging with dataset-specific formatting

### token_analysis.py
Provides detailed analysis of token probability distributions during single inference traces. Analyzes entropy patterns, token consistency, and probability landscapes to understand model behavior during generation.

## Token Probability Analysis
Explore model behavior during inference:
```bash
# Analyze token distributions for a single reasoning trace
python token_analysis.py --output_dir "analysis_results"

# Or use the convenience script
./run_token_analysis.sh
```

This generates comprehensive visualizations including:
- Entropy evolution during generation
- Top-k token probability heatmaps  
- Token consistency analysis
- Probability landscape visualization
- Individual token PDFs (one plot per generated token showing full probability distribution)

For faster analysis, skip the individual PDFs:
```bash
python token_analysis.py --output_dir "fast_analysis" --skip_individual_pdfs
```

## Soft Thinking Mode 🧠

This implementation includes an experimental "soft thinking" feature inspired by recent research on preserving information during token generation. Instead of always sampling a single token from the probability distribution (which can lose information), soft thinking:

1. **Samples top-k tokens** with their probabilities
2. **Creates weighted embeddings** by mixing the top-k token embeddings based on their probabilities  
3. **Feeds mixed embeddings** to the next layer, preserving the superposition of likely tokens
4. **Exits to normal generation** when `</reasoning>` becomes the most likely token

### Usage

Enable soft thinking mode:
```bash
# Basic soft thinking (top-2 tokens)
python main.py --soft_thinking

# Customize parameters
python main.py --soft_thinking --soft_thinking_k 3 --soft_thinking_temperature 1.2
```

### Parameters

- `--soft_thinking`: Enable soft thinking mode
- `--soft_thinking_k`: Number of top tokens to mix (default: 2)
- `--soft_thinking_temperature`: Temperature for probability mixing (default: 1.0)

### Testing

Test both generation modes:
```bash
python test_soft_thinking.py
```

This will verify that both normal and soft thinking generation work correctly and show the differences in token handling.

## Results
Training was conducted on a single H100 GPU. After ~400 training steps:

![Training Results](plots/train_score.png)

And results on the validation set - this shows a clearer sign of learning:
![Eval Results](plots/eval_score.png)

## New Features

### Mixture-of-Token Generation
The implementation now supports advanced token generation strategies:
- **Phase Transitions**: Automatic switching between mixture and normal generation modes using `</think>` token triggers
- **Multiple Sampling Strategies**:
  - Standard mixture sampling with top-k token selection
  - Nucleus sampling for dynamic token selection
  - Dirichlet sampling for probabilistic mixture weights
  - Element-wise max for deterministic selection
- **Token Embedding Tracking**: Detailed logging of token embeddings and selection probabilities

### Multi-Dataset Support
Extended beyond GSM8K to support:
- **GSM8K**: Original grade school math problems
- **Math500**: More challenging math problems
- **MBPP**: Python programming problems with safe execution
- **LeetCode**: Competitive programming problems
- **Reasoning Gym**: Various reasoning tasks (shortest path, family relationships, number sequences, maze, sokoban)

### Advanced Configuration
The training script supports numerous experimental configurations:
- `--experiment_name`: Configure different token generation strategies
- `--mixture_k`: Control number of tokens in mixture
- `--loss_on_all_tokens`: Toggle loss computation strategy
- `--normal_generation`: Fallback to standard generation for comparison

## Future Directions
I'm really pleased to see how well the key mechanics work even in this simplified implementation. Building on this, I am very excited about several directions:

1. **Self-play capabilities** where agents compete and learn from each other using relative rewards. This would create a more dynamic training environment where the reward signal comes from agent interactions rather than fixed metrics.

2. **Implementing soft reward structures**, particularly for complex reasoning tasks. I've written a framework for AI debate that I'm excited to try out.

3. **Expanding into vision-language models (VLMs)** to improve world modeling capabilities. I have an idea about using R1-style training to enhance how VLMs build and maintain internal world models that I'm really excited to explore. (Really excited about this idea - if anyone else is interested I would love to talk.)

4. **I'd like to do all this experimentation in this framework**, so I need to make things faster, and support multi-gpu training.

5. **Enhanced mixture-of-token strategies** - exploring more sophisticated token selection mechanisms and phase transition triggers.

6. **Multi-model architectures** - extending beyond Qwen to support other model families and hybrid architectures.

7. **Advanced evaluation frameworks** - developing more sophisticated reward functions and evaluation metrics for different task domains.

8. **Production optimization** - implementing model distillation, quantization, and other techniques for deployment.



