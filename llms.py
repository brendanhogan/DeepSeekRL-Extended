"""
Module for loading LLMs and their tokenizers from huggingface. 

"""
import torch
import time
import logging
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig


class InferenceModelInterface:
    """
    A unified interface for model inference with automatic retries, deterministic generation,
    and proper formatting. Supports both local HuggingFace models and OpenAI API.
    """
    
    def __init__(self, model_name: str, device: str, max_retries: int = 3):
        self.model_name = model_name
        self.device = device
        self.max_retries = max_retries
        
        # Determine if this is an OpenAI model or local model
        self.is_openai = any(api_model in model_name.lower() for api_model in ['gpt-3.5', 'gpt-4', 'openai'])
        
        if self.is_openai:
            self._setup_openai()
        else:
            self._setup_local_model()
    
    def _setup_openai(self):
        """Setup OpenAI API client"""
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.model = None
            self.tokenizer = None
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def _setup_local_model(self):
        """Setup local HuggingFace model"""
        self.model, self.tokenizer = get_llm_tokenizer(self.model_name, self.device)
        self.client = None
    
    def generate(self, prompt: str, max_tokens: int = 512, system_prompt: Optional[str] = None) -> str:
        """
        Generate text from prompt with automatic retries and deterministic settings.
        
        Args:
            prompt: The input prompt text
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt for chat models
            
        Returns:
            Generated text string
        """
        for attempt in range(self.max_retries):
            try:
                if self.is_openai:
                    return self._generate_openai(prompt, max_tokens, system_prompt)
                else:
                    return self._generate_local(prompt, max_tokens, system_prompt)
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                
                # Exponential backoff for retries
                time.sleep(2 ** attempt)
        
        raise RuntimeError("Generation failed after all retries")
    
    def _generate_openai(self, prompt: str, max_tokens: int, system_prompt: Optional[str]) -> str:
        """Generate using OpenAI API"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,  # Deterministic generation
            seed=42  # For deterministic generation
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_local(self, prompt: str, max_tokens: int, system_prompt: Optional[str]) -> str:
        """Generate using local HuggingFace model"""
        # Format as chat if system prompt provided
        if system_prompt:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = prompt
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Configure deterministic generation
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=False,  # Deterministic generation
            temperature=None,  # Not used when do_sample=False
            top_p=None,       # Not used when do_sample=False
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=generation_config
            )

        
        # Decode only the new tokens
        new_tokens = outputs[0]#[inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return generated_text.strip()


def get_llm_tokenizer(model_name: str, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None, 
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    return model, tokenizer


def get_inference_model_interface(model_name: str, device: str, max_retries: int = 3) -> InferenceModelInterface:
    """
    Factory function to create an inference model interface.
    
    Args:
        model_name: Name of the model (supports both HuggingFace and OpenAI models)
        device: Device to load the model on ('cpu' or 'cuda')
        max_retries: Maximum number of retry attempts for failed generations
        
    Returns:
        InferenceModelInterface instance
    """
    return InferenceModelInterface(model_name, device, max_retries)


if __name__ == "__main__":
    """
    Test the inference model interface with different model types
    """
    print("Testing InferenceModelInterface...")
    
    # Test parameters
    test_prompt = "What is the capital of France? Explain your answer."
    system_prompt = "You are a helpful assistant. Provide concise, accurate answers."
    
   
    # Test 2: OpenAI API (if available)
    print("\n" + "="*50)
    print("Testing OpenAI API Model (if available)")
    print("="*50)

    openai_interface = get_inference_model_interface("gpt-4.1-nano", None)
    
    print("Test 1: Basic generation")
    response1 = openai_interface.generate(test_prompt, max_tokens=100)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response1}")
    

    print("Test 1: Basic generation")
    response1 = openai_interface.generate(test_prompt, max_tokens=100)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response1}")

    # print("\nTest 2: Generation with system prompt")
    # response2 = openai_interface.generate(test_prompt, max_tokens=100, system_prompt=system_prompt)
    # print(f"System: {system_prompt}")
    # print(f"Prompt: {test_prompt}")
    # print(f"Response: {response2}")
    
