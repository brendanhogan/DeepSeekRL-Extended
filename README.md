
# DeepSeek R1 Implementation

## Motivation
I wanted to recreate DeepSeek R1's  results at a smaller scale, focusing on understanding the core mechanics by implementing everything from scratch. So this is a repo that trains Qwen1.5B on the [grade school math dataset](https://github.com/openai/grade-school-math).

This implementation heavily borrows from [Will Brown's  work](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) ([@willccbb](https://x.com/willccbb)), but restructures the code into a format optimized for learning and experimentation.

The key difference in my implementation is computing the GRPO loss function directly rather than using external RL libraries, and reformatting into a multi script repo.

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
Contains the core training loop implementing GRPO (Generalized Reward-Powered Optimization). Handles model training, evaluation, and metric tracking. 

### llms.py 
Manages model loading and configuration, currently supporting LLaMA + Qwen models through Hugging Face's transformers library. Designed to be easily extensible to other model architectures.

### rldatasets.py
Handles dataset loading and preprocessing, currently focused on GSM8K math problems. Implements custom data loaders for both training and evaluation.

### evaluator.py
Contains evaluation metrics and reward functions, closely following DeepSeek's original implementation.

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

## Future Directions
I'm really pleased to see how well the key mechanics work even in this simplified implementation. Building on this, I am very excited about several directions:

1. Adding self-play capabilities where agents compete and learn from each other using relative rewards. This would create a more dynamic training environment where the reward signal comes from agent interactions rather than fixed metrics.

2. Implementing soft reward structures, particularly for complex reasoning tasks. I've writing a framework for AI debate that I'm excited to try out.

3. Expanding into vision-language models (VLMs) to improve world modeling capabilities. I have an idea about using R1-style training to enhance how VLMs build and maintain internal world models that I'm really excited to explore. (Really excited about this idea - if anyone else is interested I would love to talk.)

4. I'd like to do all this experimentation in this framework, so I need to make things faster, and support multi-gpu training.



