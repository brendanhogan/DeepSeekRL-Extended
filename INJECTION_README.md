# Hidden State Injection for GRPO Training

## 🧠 What is Hidden State Injection?

This is an experimental extension to the GRPO implementation that learns to improve task performance by injecting learned "virtual tokens" directly into the language model's hidden state space, rather than fine-tuning the entire model.

## 🎯 Key Idea

Instead of training the entire 1.5B parameter language model, we:

1. **Freeze the LLM completely** - no parameters change
2. **Learn a tiny 3-layer neural network** (~1M parameters) 
3. **Transform the final hidden state** after processing the prompt
4. **Inject this as a "virtual token"** at the embedding level
5. **Continue generation normally** from there

Think of it as learning a continuous, task-specific "prompt" in hidden space rather than discrete text tokens.

## 🚀 Usage

### Hidden State Injection Mode
```bash
python main.py --use_hidden_injection --output_dir injection_results
```

### Standard GRPO Mode (for comparison)
```bash
python main.py --output_dir standard_results
```

## 📊 Expected Benefits

1. **Efficiency**: Only train ~1M parameters instead of ~1.5B
2. **Interpretability**: The small network becomes a learned "prompt encoder"
3. **Flexibility**: Continuous representations can capture more nuanced patterns than discrete tokens
4. **Preservation**: Base model capabilities remain completely intact

## 🔧 Architecture Details

### HiddenStateInjector Network
```
Input (hidden_size) → Linear(hidden_size//4) → ReLU → 
Linear(hidden_size) → ReLU → Linear(hidden_size) → Output
```

- **Initialization**: Starts as identity mapping for conservative training
- **Regularization**: L2 penalty to prevent drifting too far from identity
- **Training**: Standard GRPO with advantages computed from task rewards

### Generation Process
1. Process prompt normally to get final hidden state
2. Transform via injector network: `virtual_token = injector(final_hidden)`
3. Concatenate to prompt embeddings: `[prompt_embeddings, virtual_token]`
4. Continue autoregressive generation from extended context

## 📈 Key Metrics to Watch

- **Policy Loss**: How well the injector maximizes task rewards
- **Regularization Loss**: How much the injector deviates from identity
- **Advantage Mean/Std**: Quality and variance of generated solutions
- **Task Performance**: Accuracy on math problems

## 🧪 Testing

Run a quick test of both modes:
```bash
python test_injection.py
```

Or use the example experiment script:
```bash
bash run_injection_experiment.sh
```

## 🎨 Customization

The injection approach is highly modular. You can easily:

- **Change network architecture**: Modify `HiddenStateInjector` class
- **Adjust injection point**: Inject at different layers or multiple points
- **Experiment with regularization**: Try different penalties or constraints
- **Combine approaches**: Use multiple injectors or mix with other techniques

## 🤔 Why This Might Work

1. **Hidden spaces are rich**: LLM hidden states contain much more information than token embeddings
2. **Task-specific transformations**: Small networks can learn powerful, focused transformations
3. **Preserved capabilities**: No risk of catastrophic forgetting since base model is frozen
4. **Efficient optimization**: Much smaller parameter space for faster convergence

## 🔬 Research Questions

This implementation opens up several interesting research directions:

- How does performance compare to full fine-tuning?
- Can we inject at multiple layers for better results?
- What happens with different injection network architectures?
- Can we learn interpretable transformations in hidden space?
- How does this extend to other tasks beyond math reasoning?

---

**Happy experimenting!** 🎉

This is a clean, Carl Sagan-style implementation focused on understanding the core mechanics. The code prioritizes readability and interpretability over optimization tricks, making it perfect for research and experimentation. 