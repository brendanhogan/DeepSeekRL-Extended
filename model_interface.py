"""
Abstract interface for language models and their implementations.
"""
import time
import torch
import openai
import anthropic
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
from qwen_vl_utils import process_vision_info


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
    """Implementation of ModelInterface for Hugging Face models.
    
    
    Again pretty hardcoded to work for Qwen2.5 VL for now.
    
    """

    
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
    

    def generate_judge_response(self, system_prompt: str, user_prompt: str, image_path: str, image_path_2: str, **kwargs) -> str:
        """Generates a response from the model, using image inputs in the expected format with labels."""
        # Start with the main text prompt



        # Build messages of judge prompt
        user_content = [
            {"type": "text", "text": user_prompt},
        ]
        # Add SVG1 label and image if path is valid
        user_content.append({"type": "text", "text": "\n\nSVG 1:"})
        user_content.append({"type": "image", "image": image_path})

        # Add SVG2 label and image if path is valid
        user_content.append({"type": "text", "text": "\n\nSVG 2:"})
        user_content.append({"type": "image", "image": image_path_2})

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content}
        ]
        

        # Tokenize/prepare image
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.tokenizer(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)



        # Generate
        gen_kwargs = {
           "max_new_tokens": kwargs.get("max_new_tokens", 100),
           "temperature": kwargs.get("temperature", 0.1)
        }

        # Inference: Generation of the output
        outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode
        start_index = inputs["input_ids"].shape[1]
        new_tokens = outputs[0, start_index:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)


        return response.strip()

    def judge_svg_pair_conversationally(
        self,
        scene_description: str,
        image_path_1: str,
        image_path_2: str,
        max_new_tokens: int = 500, # Default max tokens for conversational turns
        temperature: float = 0.1
    ) -> Tuple[str, List[Dict]]:
        """
        Conducts a multi-turn conversation with the judge model to evaluate two SVG images.

        Args:
            scene_description: The text description of the scene.
            image_path_1: Path to the first SVG image.
            image_path_2: Path to the second SVG image.
            max_new_tokens: Max tokens for each conversational turn (except the last).
            temperature: Temperature for generation.

        Returns:
            A tuple containing:
                - final_verdict (str): The final verdict ('SVG_1_WINS' or 'SVG_2_WINS').
                - conversation_log (List[Dict]): The full conversation history.
        """
        conversation_log = []
        gen_kwargs = {"temperature": temperature} # Base generation args

        def _call_model(current_log: List[Dict]) -> str:
            """Helper function to call the model and update the log."""
            # Tokenize/prepare image(s) - Note: process_vision_info needs the *last* user message content
            prompt_text = self.tokenizer.apply_chat_template(current_log, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(current_log) # Process based on the whole log
            inputs = self.tokenizer(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # --- Use GenerationConfig ---
            generation_config = GenerationConfig(
                max_new_tokens=800,
                temperature=temperature, # Use temperature from outer scope
                # Add other parameters like top_p, top_k if needed
            )

            # Generate
            outputs = self.model.generate(**inputs, generation_config=generation_config)

            # Decode
            start_index = inputs["input_ids"].shape[1]
            new_tokens = outputs[0, start_index:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Add assistant response to log
            current_log.append({"role": "assistant", "content": response})
            return response

        # --- Turn 1: Introduction + Scene ---
        system_prompt = "You are an expert SVG evaluator. Your task is to compare two SVG images based on quality and adherence to a scene description."
        user_prompt_1 = f"I will give you two images, SVG_1 and SVG_2, generated from SVG code for a particular scene. Your ultimate job is to judge which image is higher quality. First, here is the scene description:\n\n{scene_description}"
        conversation_log.append({"role": "system", "content": system_prompt})
        conversation_log.append({"role": "user", "content": user_prompt_1})
        # _call_model(conversation_log) # Shorter response for initial ack

        # --- Turn 2: Analyze SVG_1 ---
        user_prompt_2_content = [
            {"type": "text", "text": "Okay, here is the first image, called SVG_1. Describe what you like or dislike about it."},
            {"type": "image", "image": image_path_1}
        ]
        conversation_log.append({"role": "user", "content": user_prompt_2_content})
        _call_model(conversation_log)

        # --- Turn 3: Analyze SVG_2 ---
        user_prompt_3_content = [
            {"type": "text", "text": "Okay, here is the second image, SVG_2. Describe what you like or dislike about it."},
            {"type": "image", "image": image_path_2}
        ]
        conversation_log.append({"role": "user", "content": user_prompt_3_content})
        _call_model(conversation_log)

        # --- Turn 4: Compare and Discuss ---
        user_prompt_4 = "Now, please discuss which image you think is better. Weigh the quality of the image most heavily. If the quality is roughly equal, then consider which one better adheres to the scene description."
        conversation_log.append({"role": "user", "content": user_prompt_4})
        _call_model(conversation_log)

        # --- Turn 5: Final Verdict ---
        user_prompt_5 = "Okay, based on your analysis, please respond with EXACTLY 'SVG_1_WINS' or 'SVG_2_WINS' - indicating which image you think is better. You MUST choose a winner; a tie is not allowed."
        conversation_log.append({"role": "user", "content": user_prompt_5})
        # Use fewer tokens for the final verdict to encourage a direct answer
        final_verdict_raw = _call_model(conversation_log)

        # Process final verdict
        final_verdict = final_verdict_raw.strip().upper()
        # Basic check for expected format, but return raw if not matched exactly
        # The evaluator handles non-standard verdicts.
        if "SVG_1_WINS" in final_verdict:
            final_verdict = "SVG_1_WINS"
        elif "SVG_2_WINS" in final_verdict:
             final_verdict = "SVG_2_WINS"
        # else: return the raw response

        return final_verdict, conversation_log

class OpenAIModel(ModelInterface):
    """Implementation of ModelInterface for OpenAI API models."""
    
    def __init__(self, model_name: str):
        self.client = openai.OpenAI()
        self.model_name = model_name
        
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # Translate common HF parameters to OpenAI parameters
        openai_kwargs = {}
        if 'max_new_tokens' in kwargs:
            openai_kwargs['max_tokens'] = kwargs.pop('max_new_tokens')
        if 'temperature' in kwargs:
            openai_kwargs['temperature'] = kwargs.pop('temperature')
        if 'top_p' in kwargs:
            openai_kwargs['top_p'] = kwargs.pop('top_p')
            
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
                    **openai_kwargs
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