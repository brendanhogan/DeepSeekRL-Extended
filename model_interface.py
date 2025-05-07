"""
Abstract interface for language models and their implementations.
"""
import time
import torch
import openai
import anthropic
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase

class ModelInterface(ABC):
    """Abstract base class for language model interfaces."""
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Generate text from the model given a system prompt and user prompt.
        
        Args:
            system_prompt: The system prompt/instructions
            user_prompt: The user's input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            str: The generated text
        """
        pass

class HuggingFaceModel(ModelInterface):
    """Implementation of ModelInterface for Hugging Face models."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # Format prompt in chat template
        prompt = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        prompt_text = self.tokenizer.apply_chat_template(prompt, tokenize=False)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            padding_side="left"
        ).to(self.device)
        
        # Generate
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **kwargs
        )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()

class OpenAIModel(ModelInterface):
    """Implementation of ModelInterface for OpenAI API models."""
    
    def __init__(self, model_name: str):
        self.client = openai.OpenAI()
        self.model_name = model_name
        
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # Translate common HF parameters to OpenAI parameters
        openai_kwargs = {}
        # if 'max_new_tokens' in kwargs:
        #     openai_kwargs['max_tokens'] = kwargs.pop('max_new_tokens')
        # if 'temperature' in kwargs:
        #     openai_kwargs['temperature'] = kwargs.pop('temperature')
        # if 'top_p' in kwargs:
        #     openai_kwargs['top_p'] = kwargs.pop('top_p')
            
        max_retries = 5
        base_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                    
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)  # 1, 2, 4, 8, 16 seconds
                time.sleep(delay)
                continue

class AnthropicModel(ModelInterface):
    """Implementation of ModelInterface for Anthropic API models."""
    
    def __init__(self, model_name: str):
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get('max_tokens', 4096),
            messages=[
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ]
        )
        return response.content[0].text.strip() 