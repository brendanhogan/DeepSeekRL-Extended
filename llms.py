"""
Module for loading LLMs and their tokenizers from huggingface. 

"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from model_interface import ModelInterface, HuggingFaceModel, OpenAIModel, AnthropicModel


def get_llm_tokenizer(model_name: str, device: str, is_vl: bool = False) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
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
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map=None, 
    # ).to(device)

    if is_vl:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto", 
        )
        tokenizer = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token
        model.config.pad_token_id = tokenizer.tokenizer.pad_token_id
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto", 
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
    
    return model, tokenizer

def get_judge_model(model_name: str, device: str) -> ModelInterface:
    """
    Create a judge model interface based on the model name.
    
    Args:
        model_name: Name of the model to use (can be HF model name or API model name)
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        ModelInterface: The judge model interface
    """
    if model_name.startswith(('gpt-', 'claude-')):
        if model_name.startswith('gpt-'):
            return OpenAIModel(model_name)
        else:
            return AnthropicModel(model_name)
    else:
        if "VL" in model_name:
            model, tokenizer = get_llm_tokenizer(model_name, device, is_vl=True)
            return HuggingFaceModel(model, tokenizer, device)
        else:
            model, tokenizer = get_llm_tokenizer(model_name, device)
            return HuggingFaceModel(model, tokenizer, device)

def get_compare_model(model_name: str, device: str) -> ModelInterface:
    """
    Create a compare model interface based on the model name.
    
    Args:
        model_name: Name of the model to use (can be HF model name or API model name)
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        ModelInterface: The compare model interface
    """
    if model_name.startswith(('gpt-', 'claude-')):
        if model_name.startswith('gpt-'):
            return OpenAIModel(model_name)
        else:
            return AnthropicModel(model_name)
    else:
        print("HERE") 
        print(model_name)
        print("VL" in model_name)
        if "VL" in model_name:
            model, tokenizer = get_llm_tokenizer(model_name, device, is_vl=True)
            return HuggingFaceModel(model, tokenizer, device)
        else:
            model, tokenizer = get_llm_tokenizer(model_name, device)
            return HuggingFaceModel(model, tokenizer, device)
