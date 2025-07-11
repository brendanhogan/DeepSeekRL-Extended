"""
Abstract base class and implementations for reward computation in RL training.

"""

import re
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
from model_interface import ModelInterface

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
        self.judge_prompt = """You are an impartial debate judge. Your task is to evaluate two arguments on a given topic and decide which one is more compelling.

First, you will think through the evaluation step-by-step inside a <thinking> block. Analyze each argument against the following rubric:
1.  **Logical Coherence (40%)**: Is the argument well-reasoned, logical, and free of fallacies?
2.  **Evidence and Examples (30%)**: Are claims supported by relevant evidence, examples, or data?
3.  **Clarity and Structure (20%)**: Is the argument clearly structured and easy to follow?
4.  **Persuasiveness (10%)**: How persuasive is the argument overall? Does it effectively address the core issue?

Second, after your reasoning, you will provide your final decision in an <evaluation> block. The decision must be one of the following, and nothing else:
- ARGUMENT_1_WINS
- ARGUMENT_2_WINS

A tie is not allowed.

Here is the debate:

Topic: {topic}

Argument 1:
{arg1_response}

Argument 2:
{arg2_response}

Now, provide your evaluation.
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
                    max_new_tokens=512,
                    temperature=0.1
                ).strip().upper()
                
                decision = ""
                if "<EVALUATION>" in judge_response:
                    try:
                        decision = judge_response.split("<EVALUATION>")[-1].split("</EVALUATION>")[0].strip()
                    except IndexError:
                        decision = ""

                if "ARGUMENT_1_WINS" in decision:
                    wins[i] += 1
                    losses[j] += 1
                else:
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
            "rewards/strict_format": strict_format.mean().item(),
            "rewards/soft_format": soft_format.mean().item(),
            "rewards/xml_count": xml_count.mean().item(),
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
                max_new_tokens=512,
                temperature=0.1
            ).strip().upper()
            
            decision = ""
            if "<EVALUATION>" in judge_response:
                try:
                    decision = judge_response.split("<EVALUATION>")[-1].split("</EVALUATION>")[0].strip()
                except IndexError:
                    decision = ""

            if "ARGUMENT_1_WINS" in decision:
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
            "rewards/strict_format": strict_format.mean().item(),
            "rewards/soft_format": soft_format.mean().item(), 
            "rewards/xml_count": xml_count.mean().item()
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
            "rewards/strict_format": strict_format.mean().item(),
            "rewards/soft_format": soft_format.mean().item(),
            "rewards/xml_count": xml_count.mean().item(),
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
            "rewards/strict_format": strict_format.mean().item(),
            "rewards/soft_format": soft_format.mean().item(), 
            "rewards/xml_count": xml_count.mean().item()
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
            "rewards/strict_format": strict_format.mean().item(),
            "rewards/soft_format": soft_format.mean().item(),
            "rewards/xml_count": xml_count.mean().item(),
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
            "rewards/strict_format": strict_format.mean().item(),
            "rewards/soft_format": soft_format.mean().item(), 
            "rewards/xml_count": xml_count.mean().item()
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


