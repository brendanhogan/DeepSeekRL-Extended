"""
Abstract base class and implementations for reward computation in RL training.

"""

import re
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

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
        device: str,
        tokenizer: Any = None
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
            tokenizer: Tokenizer for computing token-based rewards (optional)
            
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


def get_evaluator(name: str, use_conciseness: bool = False) -> RewardEvaluator:
    """
    Get the appropriate reward evaluator for a given task.
    
    Args:
        name: Name of the task/dataset to get evaluator for
        use_conciseness: Whether to include the conciseness reward function
        
    Returns:
        RewardEvaluator instance for the specified task
        
    Raises:
        NotImplementedError: If evaluator for given task is not implemented
    """
    if name.lower() == "gsm8k":
        return GSM8kEvaluator(use_conciseness=use_conciseness)
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")



class GSM8kEvaluator(RewardEvaluator):
    """
    Reward evaluator for the GSM8K math problem dataset.
    
    Implements reward functions for:
    - Answer correctness
    - Integer format validation
    - XML formatting (strict and soft)
    - XML tag counting
    - Conciseness (fewer unique tokens in reasoning) - optional
    """
    
    def __init__(self, use_conciseness: bool = False):
        self.use_conciseness = use_conciseness
        self.num_reward_functions = 6 if use_conciseness else 5
    
    def _extract_xml_answer(self, text: str) -> str:
        """Extract answer from XML tags."""
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    
    def _extract_xml_reasoning(self, text: str) -> str:
        """Extract reasoning from XML tags."""
        if "<reasoning>" not in text or "</reasoning>" not in text:
            return ""
        reasoning = text.split("<reasoning>")[-1]
        reasoning = reasoning.split("</reasoning>")[0]
        return reasoning.strip()
    
    def _correctness_reward(self, prompts, completions, answer) -> List[float]:
        """Reward for correct answer."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_xml_answer(r) for r in responses]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

    def _int_format_reward(self, completions) -> List[float]:
        """Reward for integer format."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted]

    def _strict_format_reward(self, completions) -> List[float]:
        """Reward for strict XML format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [bool(re.match(pattern, r)) for r in responses]
        return [0.5 if m else 0.0 for m in matches]

    def _soft_format_reward(self, completions) -> List[float]:
        """Reward for relaxed XML format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [bool(re.match(pattern, r)) for r in responses]
        return [0.5 if m else 0.0 for m in matches]

    def _xml_count_reward(self, completions) -> List[float]:
        """Reward for XML tag counting."""
        def count_xml(text: str) -> float:
            count = 0.0
            if text.count("<reasoning>\n") == 1: count += 0.125
            if text.count("\n</reasoning>\n") == 1: count += 0.125
            if text.count("\n<answer>\n") == 1:
                count += 0.125
                count -= len(text.split("\n</answer>\n")[-1])*0.001
            if text.count("\n</answer>") == 1:
                count += 0.125
                count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
            return count
            
        responses = [completion[0]["content"] for completion in completions]
        return [count_xml(r) for r in responses]

    def _conciseness_reward(self, completions, tokenizer) -> List[float]:
        """Reward for fewer unique tokens in reasoning chain."""
        if tokenizer is None:
            return [0.0] * len(completions)
            
        rewards = []
        for completion in completions:
            response = completion[0]["content"]
            reasoning = self._extract_xml_reasoning(response)
            
            if not reasoning:
                rewards.append(0.0)
                continue
            
            # Tokenize the reasoning text
            tokens = tokenizer.encode(reasoning, add_special_tokens=False)
            unique_tokens = len(set(tokens))
            
            # Scale reward: 1.5 points at 30 tokens, 0 points at 100+ tokens
            if unique_tokens <= 30:
                reward = 1.5
            elif unique_tokens >= 100:
                reward = 0.0
            else:
                # Linear scaling between 30 and 100 tokens
                reward = 1.5 * (100 - unique_tokens) / (100 - 30)
            
            rewards.append(reward)
        
        return rewards

    def _get_unique_token_counts(self, completions, tokenizer) -> List[int]:
        """Get unique token counts for reasoning chains."""
        if tokenizer is None:
            return [0] * len(completions)
            
        counts = []
        for completion in completions:
            response = completion[0]["content"]
            reasoning = self._extract_xml_reasoning(response)
            
            if not reasoning:
                counts.append(0)
                continue
            
            # Tokenize the reasoning text
            tokens = tokenizer.encode(reasoning, add_special_tokens=False)
            unique_tokens = len(set(tokens))
            counts.append(unique_tokens)
        
        return counts

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str,
        tokenizer: Any = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the given completions."""

        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)

        # Compute all reward functions
        all_scores = [
            self._correctness_reward(prompts, completions, answer),
            self._int_format_reward(completions),
            self._strict_format_reward(completions),
            self._soft_format_reward(completions),
            self._xml_count_reward(completions)
        ]
        
        # Add conciseness reward if enabled
        if self.use_conciseness:
            all_scores.append(self._conciseness_reward(completions, tokenizer))
        
        # Fill rewards tensor
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        
        # Calculate accuracy (perfect correctness score)
        correctness_scores = rewards_per_func[:, 0]  # First reward function is correctness
        num_perfect = (correctness_scores == 2.0).sum().item()
        accuracy = num_perfect / num_completions
        
        metrics = {
            "rewards/correctness_reward_func": reward_per_func[0].item(),
            "rewards/int_reward_func": reward_per_func[1].item(), 
            "rewards/strict_format_reward_func": reward_per_func[2].item(),
            "rewards/soft_format_reward_func": reward_per_func[3].item(),
            "rewards/xmlcount_reward_func": reward_per_func[4].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "accuracy": accuracy
        }
        
        # Add conciseness metric and unique token data if enabled
        if self.use_conciseness:
            metrics["rewards/conciseness_reward_func"] = reward_per_func[5].item()
            # Get unique token counts for detailed logging
            unique_token_counts = self._get_unique_token_counts(completions, tokenizer)
            metrics["unique_token_counts"] = unique_token_counts  # Per-completion counts
            metrics["avg_unique_tokens"] = sum(unique_token_counts) / len(unique_token_counts) if unique_token_counts else 0.0
        else:
            metrics["rewards/conciseness_reward_func"] = 0.0
            metrics["unique_token_counts"] = [0] * num_completions
            metrics["avg_unique_tokens"] = 0.0
        
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        breakdown = {
            'correctness': reward_scores[0].item(),
            'integer_format': reward_scores[1].item(),
            'strict_format': reward_scores[2].item(),
            'soft_format': reward_scores[3].item(),
            'xml_count': reward_scores[4].item(),
        }
        
        if self.use_conciseness:
            breakdown['conciseness'] = reward_scores[5].item()
        else:
            breakdown['conciseness'] = 0.0
            
        return breakdown
