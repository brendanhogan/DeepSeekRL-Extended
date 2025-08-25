"""
Module for loading LLMs and their tokenizers from huggingface. 

"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase


def get_llm_tokenizer(model_name: str, device: str, use_flash_attention: bool = True) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load and configure a language model and its tokenizer.

    Args:
        model_name: Name or path of the pretrained model to load
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        tuple containing:
            - The loaded language model
            - The configured tokenizer for that model
    """
    if use_flash_attention:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if "Qwen" in model_name else "eager",
            device_map="auto", 
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto", 
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    return model, tokenizer