"""
Abstract base class and implementations for reward computation in RL training.

"""

import re
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
from model_interface import ModelInterface
from PIL import Image
import numpy as np

from tqdm import tqdm

class RewardEvaluator(ABC):
    """
    Abstract base class for reward computation in RL training.
    
    This class defines the interface for reward evaluators that can be used
    to score model completions during RL training. Implement this class to
    create custom reward functions for different tasks.
    
    The main methods that need to be implemented are:
    - compute_rewards: Computes rewards for a batch of completions
    - get_reward_breakdown: Converts raw reward scores to a labeled dictionary
    """
    
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute rewards for a batch of completions.
        
        Args:
            prompts: List of prompt messages in chat format
                    [{"role": "user", "content": "..."}, ...]
            completions: List of completion messages in chat format
                        [{"role": "assistant", "content": "..."}, ...]
            answer: Ground truth answer(s) for the prompts
            device: Device to place tensors on ("cpu" or "cuda")
            
        Returns:
            rewards_per_func: Tensor of shape (num_completions, num_reward_functions)
                            containing individual reward function scores
            metrics: Dictionary of aggregated metrics including mean rewards
                    per function and total reward
        """
        pass

    @abstractmethod
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """
        Convert raw reward scores tensor to a labeled dictionary.
        
        Args:
            reward_scores: Tensor of raw scores from compute_rewards
            
        Returns:
            Dictionary mapping reward function names to their scores
        """
        pass


def get_evaluator(name: str) -> RewardEvaluator:
    """
    Get the appropriate reward evaluator for a given task.
    
    Args:
        name: Name of the task/dataset to get evaluator for
        
    Returns:
        RewardEvaluator instance for the specified task
        
    Raises:
        NotImplementedError: If evaluator for given task is not implemented
    """
    if name.lower() == "debate":
        return DebateEvaluator()
    elif name.lower() == "ld":
        return LDEvaluator()
    elif name.lower() == "chopped":
        return ChoppedEvaluator()
    elif name.lower() == 'svg':
        return SVGEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")



class DebateEvaluator(RewardEvaluator):
    """
    Reward evaluator for debate responses using two different approaches:
    1. For training: round-robin tournament scoring between generated responses
    2. For testing: head-to-head debates against the base model
    """
    
    def __init__(self):
        self.num_reward_functions = 4  # debate score + 3 format rewards
        self.judge_prompt = """You are an impartial debate judge. You will be shown two debate responses on the same topic, 
        arguing the same side (PRO or CON). Your task is to determine which argument was more compelling based on:
        1. Logical reasoning and evidence
        2. Clear structure and organization
        3. Effective use of examples
        4. Respectful tone
        5. Addressing potential counterarguments
        
        Topic: {topic}
        
        Argument 1:
        {arg1_response}
        
        Argument 2:
        {arg2_response}
        
        Which response was more compelling? Respond with EXACTLY one of these options:
        - ARGUMENT_1_WINS
        - ARGUMENT_2_WINS

        YOU MUST CHOOSE A WINNER, A TIE IS NOT ALLOWED
        """
        
    def _extract_xml_answer(self, text: str) -> str:
        """Extract the answer portion from XML tags."""
        try:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            return answer.strip()
        except:
            return text  # Fallback if format is incorrect
   
    def _strict_format_reward(self, completions) -> List[float]:
        """Reward for strict XML format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        matches = [bool(re.match(pattern, r)) for r in completions]
        return [0.5 if m else 0.0 for m in matches]

    def _soft_format_reward(self, completions) -> List[float]:
        """Reward for relaxed XML format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        matches = [bool(re.match(pattern, r)) for r in completions]
        return [0.5 if m else 0.0 for m in matches]

    def _xml_count_reward(self, completions) -> List[float]:
        """Reward for XML tag counting."""
        def count_xml(text: str) -> float:
            count = 0.0
            if "<reasoning>" in text: count += 0.125
            if "</reasoning>" in text: count += 0.125
            if "<answer>" in text: count += 0.125
            if "</answer>" in text: count += 0.125
            
            # Only penalize actual content after final tag
            if "</answer>" in text:
                count -= len(text.split("</answer>")[-1].strip())*0.001
            return count
            
        return [count_xml(r) for r in completions]
        
    def _compute_train_rewards(
        self,
        input_prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Round-robin tournament scoring for training + format rewards."""
        num_completions = len(train_model_completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Track wins/losses for each completion
        wins = torch.zeros(num_completions, device=device)
        losses = torch.zeros(num_completions, device=device)
        
        # Get debate scores using round-robin tournament
        for i in tqdm(range(num_completions), desc="Evaluating completions", leave=False):
            for j in range(i + 1, num_completions):
                topic = input_prompt.split('\nPosition:')[0].split("Debate Topic: ")[1]
                response1 = self._extract_xml_answer(train_model_completions[i])
                response2 = self._extract_xml_answer(train_model_completions[j])
                
                judge_prompt = self.judge_prompt.format(
                    topic=topic,
                    arg1_response=response1,
                    arg2_response=response2
                )
                
                # Get judge's decision using the interface
                judge_response = all_models["judge_model"].generate(
                    system_prompt="You are an impartial debate judge.",
                    user_prompt=judge_prompt,
                    max_new_tokens=50,
                    temperature=0.1
                ).strip().upper()
                
                if "ARGUMENT_1_WINS" in judge_response:
                    wins[i] += 1
                    losses[j] += 1
                elif "ARGUMENT_2_WINS" in judge_response:
                    wins[j] += 1
                    losses[i] += 1

        # Calculate normalized scores (-1.5 to 1.5 range)
        total_matches = num_completions - 1  # number of matches per completion
        win_rate = wins / total_matches
        loss_rate = losses / total_matches
        debate_scores = (win_rate - loss_rate) * 1.5  # Scale to desired range

        # Get format rewards
        strict_format = torch.tensor(
            self._strict_format_reward(train_model_completions), 
            device=device
        )
        soft_format = torch.tensor(
            self._soft_format_reward(train_model_completions), 
            device=device
        )
        xml_count = torch.tensor(
            self._xml_count_reward(train_model_completions), 
            device=device
        )
        
        # Combine all rewards
        rewards_per_func[:, 0] = debate_scores
        rewards_per_func[:, 1] = strict_format
        rewards_per_func[:, 2] = soft_format
        rewards_per_func[:, 3] = xml_count
        
        metrics = {
            "rewards/debate_score": debate_scores.mean().item(),
            "rewards/strict_format": strict_format.float().mean().item(),
            "rewards/soft_format": soft_format.float().mean().item(),
            "rewards/xml_count": xml_count.float().mean().item(),
            "reward": rewards_per_func.sum(dim=1).mean().item()
        }
        
        return rewards_per_func, metrics

    def _compute_test_rewards(
        self,
        prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: List[str],
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Head-to-head debates against base model for testing."""
        num_debates = len(train_model_completions)
        rewards_per_func = torch.zeros(num_debates, self.num_reward_functions, device=device)
        wins = 0
        
        # Get format rewards
        strict_format = torch.tensor(
            self._strict_format_reward(train_model_completions), 
            device=device
        )
        soft_format = torch.tensor(
            self._soft_format_reward(train_model_completions), 
            device=device
        )
        xml_count = torch.tensor(
            self._xml_count_reward(train_model_completions), 
            device=device
        )
        
        topic = prompt.split('\nPosition:')[0].split("Debate Topic: ")[1]
        
        for i in range(num_debates):
            # Get trained model's response
            trained_response = self._extract_xml_answer(train_model_completions[i])
            
            # Get compare model's response
            compare_response = self._extract_xml_answer(compare_model_completions[i])     

            # Format judge prompt
            judge_prompt = self.judge_prompt.format(
                topic=topic,
                arg1_response=trained_response,
                arg2_response=compare_response
            )
            
            # Get judge's decision using the interface
            judge_response = all_models["judge_model"].generate(
                system_prompt="You are an impartial debate judge.",
                user_prompt=judge_prompt,
                max_new_tokens=50,
                temperature=0.1
            ).strip().upper()
            
            if "ARGUMENT_1_WINS" in judge_response:
                score = 1.0
                rewards_per_func[i, 0] = score
                wins += 1

            # Add format rewards
            rewards_per_func[i, 1] = strict_format[i]
            rewards_per_func[i, 2] = soft_format[i]
            rewards_per_func[i, 3] = xml_count[i]

        win_rate = wins / num_debates
        metrics = {
            "win_rate": win_rate,
            "reward": rewards_per_func.mean().item(),
            "num_wins": wins,
            "num_debates": num_debates,
            "rewards/strict_format": strict_format.float().mean().item(),
            "rewards/soft_format": soft_format.float().mean().item(), 
            "rewards/xml_count": xml_count.float().mean().item()
        }
        
        return rewards_per_func, metrics

    def compute_rewards(
        self,
        input_prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: Optional[List[str]] = None,
        device: str = "cuda",
        is_test: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute rewards - different behavior for training vs testing."""
        if is_test:
            if compare_model_completions is None:
                 raise ValueError("compare_model_completions must be provided when is_test=True")
            return self._compute_test_rewards(input_prompt, all_models, train_model_completions, compare_model_completions, device)
        else:
            return self._compute_train_rewards(input_prompt, all_models, train_model_completions, device)
            
    def get_reward_breakdown(self, rewards: torch.Tensor) -> Dict[str, float]:
        """Convert raw reward scores to a labeled dictionary."""
        return {
            "debate_score": rewards[0].item(),
            "strict_format": rewards[1].item(),
            "soft_format": rewards[2].item(),
            "xml_count": rewards[3].item()
        }


class LDEvaluator(RewardEvaluator):
    """
    Reward evaluator for Larry David-style roasts using two different approaches:
    1. For training: round-robin tournament scoring between generated responses
    2. For testing: head-to-head comparisons against the base model
    """
    
    def __init__(self):
        self.num_reward_functions = 4  # humor score + 3 format rewards
        self.judge_prompt = """You are a comedy judge. You will be shown two comedy bits in the style of Larry David making fun of something.

        Your only job is to pick which one is funnier. Two critical rules:

        1. Pick the funniest bit regardless of length or structure - a messy, extremely short bit could win if it's funnier
        2. AUTOMATIC LOSS: If a bit doesn't speak in first person AS Larry David (e.g. if it says "this is what Larry David might say" or describes what he would say)

        Subject: {topic}

        Bit 1:
        {arg1_response}

        Bit 2:
        {arg2_response}

        Which bit was funnier? Respond with EXACTLY one of these options:
        - BIT_1_WINS
        - BIT_2_WINS

        YOU MUST CHOOSE A WINNER, A TIE IS NOT ALLOWED
        Remember: Any bit that doesn't speak AS Larry David in first person automatically loses. For bits that both speak as Larry, pick the funnier one regardless of length or structure."""

        
    def _extract_xml_answer(self, text: str) -> str:
        """Extract the answer portion from XML tags."""
        try:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            return answer.strip()
        except:
            return text  # Fallback if format is incorrect
   
    def _strict_format_reward(self, completions) -> List[float]:
        """Reward for strict XML format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        matches = [bool(re.match(pattern, r)) for r in completions]
        return [0.5 if m else 0.0 for m in matches]

    def _soft_format_reward(self, completions) -> List[float]:
        """Reward for relaxed XML format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        matches = [bool(re.match(pattern, r)) for r in completions]
        return [0.5 if m else 0.0 for m in matches]

    def _xml_count_reward(self, completions) -> List[float]:
        """Reward for XML tag counting."""
        def count_xml(text: str) -> float:
            count = 0.0
            if "<reasoning>" in text: count += 0.125
            if "</reasoning>" in text: count += 0.125
            if "<answer>" in text: count += 0.125
            if "</answer>" in text: count += 0.125
            
            # Only penalize actual content after final tag
            if "</answer>" in text:
                count -= len(text.split("</answer>")[-1].strip())*0.001
            return count
            
        return [count_xml(r) for r in completions]
        
    def _compute_train_rewards(
        self,
        input_prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Round-robin tournament scoring for training + format rewards."""
        num_completions = len(train_model_completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Track wins/losses for each completion
        wins = torch.zeros(num_completions, device=device)
        losses = torch.zeros(num_completions, device=device)
        
        # Get humor scores using round-robin tournament
        for i in tqdm(range(num_completions), desc="Evaluating completions", leave=False):
            for j in range(i + 1, num_completions):
                topic = input_prompt.split("Roast Subject: ")[1]
                response1 = self._extract_xml_answer(train_model_completions[i])
                response2 = self._extract_xml_answer(train_model_completions[j])
                
                judge_prompt = self.judge_prompt.format(
                    topic=topic,
                    arg1_response=response1,
                    arg2_response=response2
                )
                
                # Get judge's decision using the interface
                judge_response = all_models["judge_model"].generate(
                    system_prompt="You are a comedy judge specializing in Larry David's style of humor.",
                    user_prompt=judge_prompt,
                    max_new_tokens=50,
                    temperature=0.1
                ).strip().upper()
                
                if "BIT_1_WINS" in judge_response:
                    wins[i] += 1
                    losses[j] += 1
                else:
                    wins[j] += 1
                    losses[i] += 1

        # Calculate normalized scores (-1.5 to 1.5 range)
        total_matches = num_completions - 1
        win_rate = wins / total_matches
        loss_rate = losses / total_matches
        humor_scores = (win_rate - loss_rate) * 1.5  # Scale to desired range

        # Get format rewards
        strict_format = torch.tensor(
            self._strict_format_reward(train_model_completions), 
            device=device
        )
        soft_format = torch.tensor(
            self._soft_format_reward(train_model_completions), 
            device=device
        )
        xml_count = torch.tensor(
            self._xml_count_reward(train_model_completions), 
            device=device
        )
        
        # Combine all rewards
        rewards_per_func[:, 0] = humor_scores
        rewards_per_func[:, 1] = strict_format
        rewards_per_func[:, 2] = soft_format
        rewards_per_func[:, 3] = xml_count
        
        metrics = {
            "rewards/humor_score": humor_scores.mean().item(),
            "rewards/strict_format": strict_format.float().mean().item(),
            "rewards/soft_format": soft_format.float().mean().item(),
            "rewards/xml_count": xml_count.float().mean().item(),
            "reward": rewards_per_func.sum(dim=1).mean().item()
        }
        
        return rewards_per_func, metrics

    def _compute_test_rewards(
        self,
        prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: List[str],
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Head-to-head comparisons against base model for testing."""
        num_comparisons = len(train_model_completions)
        rewards_per_func = torch.zeros(num_comparisons, self.num_reward_functions, device=device)
        wins = 0
        
        # Get format rewards
        strict_format = torch.tensor(
            self._strict_format_reward(train_model_completions), 
            device=device
        )
        soft_format = torch.tensor(
            self._soft_format_reward(train_model_completions), 
            device=device
        )
        xml_count = torch.tensor(
            self._xml_count_reward(train_model_completions), 
            device=device
        )
        
        topic = prompt.split("Roast Subject: ")[1]
        
        for i in range(num_comparisons):
            # Get trained model's response
            trained_response = self._extract_xml_answer(train_model_completions[i])
            
            # Get compare model's response
            compare_response = self._extract_xml_answer(compare_model_completions[i])     

            # Format judge prompt
            judge_prompt = self.judge_prompt.format(
                topic=topic,
                arg1_response=trained_response,
                arg2_response=compare_response
            )
            
            # Get judge's decision using the interface
            judge_response = all_models["judge_model"].generate(
                system_prompt="You are a comedy judge specializing in Larry David's style of humor.",
                user_prompt=judge_prompt,
                max_new_tokens=50,
                temperature=0.1
            ).strip().upper()
            
            if "BIT_1_WINS" in judge_response:
                score = 1.0
                rewards_per_func[i, 0] = score
                wins += 1

            # Add format rewards
            rewards_per_func[i, 1] = strict_format[i]
            rewards_per_func[i, 2] = soft_format[i]
            rewards_per_func[i, 3] = xml_count[i]

        win_rate = wins / num_comparisons
        metrics = {
            "win_rate": win_rate,
            "reward": rewards_per_func.mean().item(),
            "num_wins": wins,
            "num_comparisons": num_comparisons,
            "rewards/strict_format": strict_format.float().mean().item(),
            "rewards/soft_format": soft_format.float().mean().item(), 
            "rewards/xml_count": xml_count.float().mean().item()
        }
        
        return rewards_per_func, metrics

    def compute_rewards(
        self,
        input_prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: Optional[List[str]] = None,
        device: str = "cuda",
        is_test: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute rewards - different behavior for training vs testing."""
        if is_test:
            if compare_model_completions is None:
                 raise ValueError("compare_model_completions must be provided when is_test=True")
            return self._compute_test_rewards(input_prompt, all_models, train_model_completions, compare_model_completions, device)
        else:
            return self._compute_train_rewards(input_prompt, all_models, train_model_completions, device)
            
    def get_reward_breakdown(self, rewards: torch.Tensor) -> Dict[str, float]:
        """Convert raw reward scores to a labeled dictionary."""
        return {
            "humor_score": rewards[0].item(),
            "strict_format": rewards[1].item(),
            "soft_format": rewards[2].item(),
            "xml_count": rewards[3].item()
        }


class ChoppedEvaluator(RewardEvaluator):
    """
    Reward evaluator for Chopped-style recipe generation using two different approaches:
    1. For training: round-robin tournament scoring between generated recipes
    2. For testing: head-to-head comparisons against the base model
    """
    
    def __init__(self):
        self.num_reward_functions = 4  # recipe score + 3 format rewards
        self.judge_prompt = """You are a Chopped judge evaluating two recipes that use the same mystery basket ingredients.
        Your task is to determine which recipe would taste better based on:
        1. Flavor balance and harmony
        2. Creative use of mystery ingredients
        3. Technical execution and timing
        4. Overall appeal and presentation
        5. Practicality and replicability

        Mystery Basket:
        {basket}

        Recipe 1:
        {recipe1}

        Recipe 2:
        {recipe2}

        Which recipe would taste better? Respond with EXACTLY one of these options:
        - RECIPE_1_WINS
        - RECIPE_2_WINS

        YOU MUST CHOOSE A WINNER, A TIE IS NOT ALLOWED
        Focus purely on which recipe would taste better and make better use of the mystery ingredients.
        """
        
    def _extract_xml_answer(self, text: str) -> str:
        """Extract the answer portion from XML tags."""
        try:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            return answer.strip()
        except:
            return text  # Fallback if format is incorrect
   
    def _strict_format_reward(self, completions) -> List[float]:
        """Reward for strict XML format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        matches = [bool(re.match(pattern, r)) for r in completions]
        return [0.5 if m else 0.0 for m in matches]

    def _soft_format_reward(self, completions) -> List[float]:
        """Reward for relaxed XML format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        matches = [bool(re.match(pattern, r)) for r in completions]
        return [0.5 if m else 0.0 for m in matches]

    def _xml_count_reward(self, completions) -> List[float]:
        """Reward for XML tag counting."""
        def count_xml(text: str) -> float:
            count = 0.0
            if "<reasoning>" in text: count += 0.125
            if "</reasoning>" in text: count += 0.125
            if "<answer>" in text: count += 0.125
            if "</answer>" in text: count += 0.125
            
            # Only penalize actual content after final tag
            if "</answer>" in text:
                count -= len(text.split("</answer>")[-1].strip())*0.001
            return count
            
        return [count_xml(r) for r in completions]
        
    def _compute_train_rewards(
        self,
        input_prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Round-robin tournament scoring for training + format rewards."""
        num_completions = len(train_model_completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Track wins/losses for each completion
        wins = torch.zeros(num_completions, device=device)
        losses = torch.zeros(num_completions, device=device)
        
        # Get recipe scores using round-robin tournament
        for i in tqdm(range(num_completions), desc="Evaluating completions", leave=False):
            for j in range(i + 1, num_completions):
                basket = input_prompt.split("Mystery Basket:\n")[1].strip()
                recipe1 = self._extract_xml_answer(train_model_completions[i])
                recipe2 = self._extract_xml_answer(train_model_completions[j])
                
                judge_prompt = self.judge_prompt.format(
                    basket=basket,
                    recipe1=recipe1,
                    recipe2=recipe2
                )
                
                # Get judge's decision using the interface
                judge_response = all_models["judge_model"].generate(
                    system_prompt="You are a Chopped judge evaluating recipes.",
                    user_prompt=judge_prompt,
                    max_new_tokens=50,
                    temperature=0.1
                ).strip().upper()
                
                if "RECIPE_1_WINS" in judge_response:
                    wins[i] += 1
                    losses[j] += 1
                else:
                    wins[j] += 1
                    losses[i] += 1

        # Calculate normalized scores (-1.5 to 1.5 range)
        total_matches = num_completions - 1
        win_rate = wins / total_matches
        loss_rate = losses / total_matches
        recipe_scores = (win_rate - loss_rate) * 1.5  # Scale to desired range

        # Get format rewards
        strict_format = torch.tensor(
            self._strict_format_reward(train_model_completions), 
            device=device
        )
        soft_format = torch.tensor(
            self._soft_format_reward(train_model_completions), 
            device=device
        )
        xml_count = torch.tensor(
            self._xml_count_reward(train_model_completions), 
            device=device
        )
        
        # Combine all rewards
        rewards_per_func[:, 0] = recipe_scores
        rewards_per_func[:, 1] = strict_format
        rewards_per_func[:, 2] = soft_format
        rewards_per_func[:, 3] = xml_count
        
        metrics = {
            "rewards/recipe_score": recipe_scores.mean().item(),
            "rewards/strict_format": strict_format.float().mean().item(),
            "rewards/soft_format": soft_format.float().mean().item(),
            "rewards/xml_count": xml_count.float().mean().item(),
            "reward": rewards_per_func.sum(dim=1).mean().item()
        }
        
        return rewards_per_func, metrics

    def _compute_test_rewards(
        self,
        prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: List[str],
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Head-to-head comparisons against base model for testing."""
        num_comparisons = len(train_model_completions)
        rewards_per_func = torch.zeros(num_comparisons, self.num_reward_functions, device=device)
        wins = 0
        
        # Get format rewards
        strict_format = torch.tensor(
            self._strict_format_reward(train_model_completions), 
            device=device
        )
        soft_format = torch.tensor(
            self._soft_format_reward(train_model_completions), 
            device=device
        )
        xml_count = torch.tensor(
            self._xml_count_reward(train_model_completions), 
            device=device
        )
        
        basket = prompt.split("Mystery Basket:\n")[1].strip()
        
        for i in range(num_comparisons):
            # Get trained model's response
            trained_response = self._extract_xml_answer(train_model_completions[i])
            
            # Get compare model's response
            compare_response = self._extract_xml_answer(compare_model_completions[i])     

            # Format judge prompt
            judge_prompt = self.judge_prompt.format(
                basket=basket,
                recipe1=trained_response,
                recipe2=compare_response
            )
            
            # Get judge's decision using the interface
            judge_response = all_models["judge_model"].generate(
                system_prompt="You are a Chopped judge evaluating recipes.",
                user_prompt=judge_prompt,
                max_new_tokens=50,
                temperature=0.1
            ).strip().upper()
            
            if "RECIPE_1_WINS" in judge_response:
                score = 1.0
                rewards_per_func[i, 0] = score
                wins += 1

            # Add format rewards
            rewards_per_func[i, 1] = strict_format[i]
            rewards_per_func[i, 2] = soft_format[i]
            rewards_per_func[i, 3] = xml_count[i]

        win_rate = wins / num_comparisons
        metrics = {
            "win_rate": win_rate,
            "reward": rewards_per_func.mean().item(),
            "num_wins": wins,
            "num_comparisons": num_comparisons,
            "rewards/strict_format": strict_format.float().mean().item(),
            "rewards/soft_format": soft_format.float().mean().item(), 
            "rewards/xml_count": xml_count.float().mean().item()
        }
        
        return rewards_per_func, metrics

    def compute_rewards(
        self,
        input_prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: Optional[List[str]] = None,
        device: str = "cuda",
        is_test: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute rewards - different behavior for training vs testing."""
        if is_test:
            if compare_model_completions is None:
                 raise ValueError("compare_model_completions must be provided when is_test=True")
            return self._compute_test_rewards(input_prompt, all_models, train_model_completions, compare_model_completions, device)
        else:
            return self._compute_train_rewards(input_prompt, all_models, train_model_completions, device)
            
    def get_reward_breakdown(self, rewards: torch.Tensor) -> Dict[str, float]:
        """Convert raw reward scores to a labeled dictionary."""
        return {
            "recipe_score": rewards[0].item(),
            "strict_format": rewards[1].item(),
            "soft_format": rewards[2].item(),
            "xml_count": rewards[3].item()
        }


class SVGEvaluator(RewardEvaluator):
    """

    So - this will be pretty hardcoded to work for Qwen2.5 VL for now. 


    Reward evaluator for SVG generation responses.
    Uses a judge model to compare SVG quality and adherence to prompt via a conversational approach.
    Includes format rewards for XML structure.
    """

    def __init__(self):
        self.num_reward_functions = 4  # svg_score + 3 format rewards
        # The detailed multi-turn prompt logic is now handled within the judge model's method.

    def _extract_xml_answer(self, text: str) -> str:
        """Extract the answer portion from XML tags."""
        try:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            return answer.strip()
        except:
            return "" # Return empty string if format is incorrect or answer is missing

    def _is_valid_svg_code(self, svg_code: str) -> bool:
        """Checks if the extracted answer looks like potentially valid SVG code."""
        # Simple check: must not be empty and must start with <svg (case-insensitive)
        return bool(svg_code) and svg_code.lower().startswith("<svg")

    def _strict_format_reward(self, completions) -> List[float]:
        """Reward for strict XML format."""
        # Reuse the same format reward logic
        # Note: Using re.DOTALL to match newlines within reasoning/answer
        pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
        matches = [bool(re.match(pattern, c, re.DOTALL)) for c in completions]
        return [0.5 if m else 0.0 for m in matches]

    def _soft_format_reward(self, completions) -> List[float]:
        """Reward for relaxed XML format."""
        # Reuse the same format reward logic
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        # Use re.search to find the pattern anywhere, and re.DOTALL to match across newlines
        matches = [bool(re.search(pattern, c, re.DOTALL)) for c in completions]
        return [0.5 if m else 0.0 for m in matches]

    def _xml_count_reward(self, completions) -> List[float]:
        """Reward for XML tag counting."""
        # Reuse the same format reward logic
        def count_xml(text: str) -> float:
            count = 0.0
            if "<reasoning>" in text: count += 0.125
            if "</reasoning>" in text: count += 0.125
            if "<answer>" in text: count += 0.125
            if "</answer>" in text: count += 0.125
            # Only penalize actual content after final tag
            if "</answer>" in text:
                # Ensure split doesn't fail if tag isn't present
                parts = text.split("</answer>", 1)
                if len(parts) > 1:
                    count -= len(parts[1].strip()) * 0.001
            return max(0, count) # Ensure reward doesn't go negative

        return [count_xml(c) for c in completions]

    def _compute_train_rewards(
        self,
        input_prompt: str, # Here, input_prompt is the scene description
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        train_model_image_paths: List[str], # Add image paths input
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float], List[Dict[str, Any]], List[int]]: # Adjusted return type for pairwise_results
        """Round-robin tournament scoring for training using image comparisons + format rewards."""
        num_completions = len(train_model_completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)

        # Track wins/losses for each completion
        wins = torch.zeros(num_completions, device=device)
        losses = torch.zeros(num_completions, device=device)

        scene_description = input_prompt # The prompt passed is the scene description
        judge_model = all_models["judge_model"] # Get the judge model instance

        def is_generated(img_path):
            img = Image.open(img_path).convert('RGB')
            img_data = np.array(img)
            return np.all(img_data == [0, 0, 0])

        # Get SVG scores using round-robin tournament with image comparisons
        pairwise_results = [] # Initialize pairwise results list
        for i in tqdm(range(num_completions), desc="Evaluating SVG completions (train)", leave=False):
            image_path_1 = train_model_image_paths[i]

            for j in range(i + 1, num_completions):
                image_path_2 = train_model_image_paths[j]
                winner = None
                winner_idx = None # Track the index of the winner
                final_verdict = "PRE_CHECK_FAILED" # Default if pre-check handles it
                conversation_log = [] # Initialize empty log

                # --- Black Image Pre-Check --- 
                img1_is_black = is_generated(image_path_1)
                img2_is_black = is_generated(image_path_2)

                if img1_is_black and not img2_is_black:
                    final_verdict = "SVG_2_WINS (Auto - Img1 not generated)"
                    winner = 2
                    winner_idx = j
                elif not img1_is_black and img2_is_black:
                    final_verdict = "SVG_1_WINS (Auto - Img2 not generated)"
                    winner = 1
                    winner_idx = i
                elif img1_is_black and img2_is_black:
                    final_verdict = "TIE_BOTH_NOT_GENERATED"
                    winner = 3 # Treat as a tie
                    winner_idx = None
                else:
                    final_verdict, conversation_log = judge_model.judge_svg_pair_conversationally(
                        scene_description=scene_description,
                        image_path_1=image_path_1,
                        image_path_2=image_path_2,
                        max_new_tokens=500, # Increased tokens for conversation
                        temperature=0.1
                    )
                    final_verdict = final_verdict.strip().upper() # Ensure consistent format
    

                # Determine winner based on the final verdict string
                if "SVG_1_WINS" in final_verdict:
                    winner = 1
                    winner_idx = i # Store index of winner
                elif "SVG_2_WINS" in final_verdict:
                    winner = 2
                    winner_idx = j # Store index of winner
                else:
                    winner = 3 
                    winner_idx = None
                # else: no winner decided by judge, treat as tie for scoring

                # Update wins/losses based on winner
                if winner == 1:
                    wins[i] += 1
                    losses[j] += 1
                elif winner == 2:
                    wins[j] += 1
                    losses[i] += 1
                elif winner == 3:
                    wins[i] += 0.5
                    losses[j] += 0.5
                    wins[j] += 0.5
                    losses[i] += 0.5
                # If winner is None (tie), no change to wins/losses

                # Store comparison result
                pairwise_results.append({
                    'comp_1_idx': i,
                    'comp_2_idx': j,
                    'winner_idx': winner_idx, # Store winner index or None for tie/error
                    'final_verdict': final_verdict, # Store the final verdict string
                    'conversation_log': conversation_log # Store the full conversation log
                })

        # Calculate normalized scores (-1.5 to 1.5 range)
        total_matches = num_completions - 1
        # Avoid division by zero if only one completion
        total_matches =  num_completions-1
        win_rate = wins / total_matches
        loss_rate = losses / total_matches
        svg_scores = (win_rate - loss_rate) * 1.5  # Scale to desired range

        # Get format rewards (remains the same)
        strict_format = torch.tensor(self._strict_format_reward(train_model_completions), device=device)
        soft_format = torch.tensor(self._soft_format_reward(train_model_completions), device=device)
        xml_count = torch.tensor(self._xml_count_reward(train_model_completions), device=device)

        # Combine all rewards
        rewards_per_func[:, 0] = svg_scores
        rewards_per_func[:, 1] = strict_format
        rewards_per_func[:, 2] = soft_format
        rewards_per_func[:, 3] = xml_count

        metrics = {
            "rewards/svg_score": svg_scores.mean().item(),
            "rewards/strict_format": strict_format.float().mean().item(),
            "rewards/soft_format": soft_format.float().mean().item(),
            "rewards/xml_count": xml_count.float().mean().item(),
            "reward": rewards_per_func.sum(dim=1).mean().item()
        }

        return rewards_per_func, metrics, pairwise_results, wins.tolist() # Return pairwise results and wins as a list

    def _compute_test_rewards(
        self,
        prompt: str, # Scene description
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: List[str],
        train_model_image_paths: List[str],
        compare_model_image_paths: List[str],
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, Dict[str, float]]: # Removed List return, test doesn't return pairwise results
        """Head-to-head comparisons against base model for testing using conversational judging."""
        num_comparisons = len(train_model_completions)
        rewards_per_func = torch.zeros(num_comparisons, self.num_reward_functions, device=device)
        wins = 0

        scene_description = prompt
        judge_model = all_models["judge_model"]

        # Get format rewards first
        strict_format = torch.tensor(self._strict_format_reward(train_model_completions), device=device)
        soft_format = torch.tensor(self._soft_format_reward(train_model_completions), device=device)
        xml_count = torch.tensor(self._xml_count_reward(train_model_completions), device=device)

        def is_black_image(img_path): # Replicated helper function
            try:
                img = Image.open(img_path).convert('RGB')
                img_data = np.array(img)
                return np.all(img_data <= 5)
            except Exception as e:
                print(f"Warning: Could not open or process image {img_path}: {e}")
                return True

        test_results_summary = [] # Optional: Store summary of test verdicts

        for i in range(num_comparisons):
            trained_image_path = train_model_image_paths[i]
            compare_image_path = compare_model_image_paths[i]
            winner = None
            final_verdict = "PRE_CHECK_FAILED"
            conversation_log = [] # Log is generated but not stored long-term in test by default

            # --- Black Image Pre-Check ---
            trained_img_failed = is_black_image(trained_image_path)
            compare_img_failed = is_black_image(compare_image_path)

            if trained_img_failed and not compare_img_failed:
                final_verdict = "SVG_2_WINS (Auto - Trained Img failed check)"
                winner = 2 # Compare model wins
            elif not trained_img_failed and compare_img_failed:
                final_verdict = "SVG_1_WINS (Auto - Compare Img failed check)"
                winner = 1 # Trained model wins
            elif trained_img_failed and compare_img_failed:
                final_verdict = "TIE_BOTH_FAILED"
                winner = None # Tie
            else:
                # Call the conversational judging method
                try:
                    final_verdict, conversation_log = judge_model.judge_svg_pair_conversationally(
                        scene_description=scene_description,
                        image_path_1=trained_image_path, # SVG_1 is the trained model
                        image_path_2=compare_image_path, # SVG_2 is the compare model
                        max_new_tokens=500,
                        temperature=0.1
                    )
                    final_verdict = final_verdict.strip().upper()
                except AttributeError:
                    raise NotImplementedError("The judge model does not have the required 'judge_svg_pair_conversationally' method.")
                except Exception as e:
                    print(f"Error during conversational judging for test comparison {i}: {e}")
                    final_verdict = "JUDGING_ERROR"

                # Determine winner based on the final verdict string
                if "SVG_1_WINS" in final_verdict: # Trained model wins
                    winner = 1
                elif "SVG_2_WINS" in final_verdict: # Compare model wins
                    winner = 2
                # else: winner remains None (tie/error)

            # Assign score based on winner (1.0 if trained model wins, 0.0 otherwise)
            score = 0.0
            if winner == 1:
                score = 1.0
                wins += 1
            # No score added if compare model wins (winner==2) or it's a tie/error (winner==None)

            rewards_per_func[i, 0] = score # SVG score is 1.0 for win, 0.0 otherwise
            rewards_per_func[i, 1] = strict_format[i]
            rewards_per_func[i, 2] = soft_format[i]
            rewards_per_func[i, 3] = xml_count[i]

            test_results_summary.append({'trained_idx': i, 'final_verdict': final_verdict, 'winner': winner})

        win_rate = wins / num_comparisons if num_comparisons > 0 else 0.0
        metrics = {
            "win_rate": win_rate,
            "reward": rewards_per_func.sum(dim=1).mean().item(), # Mean total reward across all functions
            "num_wins": wins,
            "num_comparisons": num_comparisons,
            "rewards/strict_format": strict_format.float().mean().item(),
            "rewards/soft_format": soft_format.float().mean().item(),
            "rewards/xml_count": xml_count.float().mean().item(),
            # "test_results_summary": test_results_summary # Optional: include summary if needed
        }

        # Note: _compute_test_rewards does not return pairwise_results or conversation logs by default
        return rewards_per_func, metrics

    def compute_rewards(
        self,
        input_prompt: str, # Expecting scene description string
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: Optional[List[str]] = None,
        train_model_image_paths: Optional[List[str]] = None,
        compare_model_image_paths: Optional[List[str]] = None,
        device: str = "cuda",
        is_test: bool = False
    # Adjusted return type annotation to reflect changes
    ) -> Tuple[torch.Tensor, Dict[str, float], Optional[List[Dict[str, Any]]], Optional[List[int]]]:
        """Compute rewards - different behavior for training vs testing."""
        # Ensure input_prompt is a string
        if not isinstance(input_prompt, str):
             raise TypeError(f"SVGEvaluator expects input_prompt to be a string (scene description), got {type(input_prompt)}")

        if is_test:
            # Test mode does not generate pairwise results, return None
            rewards_per_func, metrics = self._compute_test_rewards(input_prompt, all_models, train_model_completions, compare_model_completions, train_model_image_paths, compare_model_image_paths, device)
            return rewards_per_func, metrics
        else:
            # Train mode returns pairwise results and wins list
            rewards_per_func, metrics, pairwise_results, wins_list = self._compute_train_rewards(input_prompt, all_models, train_model_completions, train_model_image_paths, device)
            return rewards_per_func, metrics, pairwise_results, wins_list

    def get_reward_breakdown(self, rewards: torch.Tensor) -> Dict[str, float]:
        """Convert raw reward scores to a labeled dictionary."""
        # Ensure tensor has the expected shape
        if rewards.ndim == 1 and rewards.shape[0] == self.num_reward_functions:
            return {
                "svg_score": rewards[0].item(),
                "strict_format": rewards[1].item(),
                "soft_format": rewards[2].item(),
                "xml_count": rewards[3].item()
            }
        else:
            # Handle potential shape mismatch or return default/error
            print(f"Warning: Unexpected reward tensor shape in get_reward_breakdown: {rewards.shape}")
            return {
                "svg_score": 0.0,
                "strict_format": 0.0,
                "soft_format": 0.0,
                "xml_count": 0.0
            }


