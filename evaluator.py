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
        self.judge_prompt = """You are a comedy judge specializing in Larry David's style of humor. You will be shown two roasts/commentaries on the same subject. 
        Your task is to determine which one is funnier based on:
        1. Creativity and originality 
        2. Larry David-style observational humor
        3. Social commentary and relatability

        Important: Length does NOT equal quality. A shorter, punchier roast is often better than a longer one. Focus purely on how funny the content is, regardless of length.
        
        Subject: {topic}
        
        Roast 1:
        {arg1_response}
        
        Roast 2:
        {arg2_response}
        
        Which roast was funnier? Respond with EXACTLY one of these options:
        - ARGUMENT_1_WINS  
        - ARGUMENT_2_WINS

        YOU MUST CHOOSE A WINNER, A TIE IS NOT ALLOWED
        Remember: Pick the funniest roast regardless of length - shorter can be better if it's funnier!
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
                
                if "ARGUMENT_1_WINS" in judge_response:
                    wins[i] += 1
                    losses[j] += 1
                elif "ARGUMENT_2_WINS" in judge_response:
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
            
            if "ARGUMENT_1_WINS" in judge_response:
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


