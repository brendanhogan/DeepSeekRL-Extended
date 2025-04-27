"""
Abstract base class and implementations for reward computation in RL training.

"""

import re
import torch
import json
import random
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
    2. For testing: head-to-head debates against the base model using GRM-style evaluation
    """
    
    def __init__(self):
        self.num_reward_functions = 4  # debate score + 3 format rewards
        self.judge_prompt_old = """You are an impartial debate judge. You will be shown two debate responses on the same topic, 
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
        # Back to regular string, simplify example JSON escaping
        self.principles_prompt_template = """You are an impartial debate judge tasked with generating evaluation principles. 
Given the debate topic and two arguments (representing the same side), generate exactly {num_principles} distinct principles 
that would be most useful for evaluating the quality and persuasiveness of these specific arguments. Focus on aspects like 
logical soundness, evidence use, clarity, addressing counterarguments, and overall impact.

Debate Topic: {topic}

Argument 1:
{arg1_response}

Argument 2:
{arg2_response}

Based on the topic and the content of these two arguments, provide {num_principles} evaluation principles. 
Examples of principles:
- "Assess the logical flow and coherence of the reasoning."
- "Evaluate the quality and relevance of the evidence presented."
- "Consider the clarity and conciseness of the language used."
- "Check if the argument effectively anticipates and addresses potential counterarguments."
- "Gauge the overall persuasiveness and impact of the argument."

Output ONLY a JSON object containing a single key "principles" which maps to a list of strings. Example format:
{{'principles': ['Principle 1 text...', 'Principle 2 text...', ...]}}
"""
        # Simplify example JSON escaping here too
        self.critique_prompt_template = """You are an impartial debate judge providing critiques based on predefined principles. 
Given the debate topic, a specific argument, and a set of evaluation principles, you must evaluate the argument AGAINST EACH PRINCIPLE INDIVIDUALLY.

For each principle provided, output:
1. A concise (1-2 sentence) critique of the argument focusing ONLY on how well it adheres to THAT specific principle.
2. A numerical score from 1 (poor) to 10 (excellent) reflecting the argument's quality based ONLY on THAT specific principle.

Debate Topic: {topic}

Argument to Evaluate:
{argument}

Evaluation Principles:
{principles_list_str}
# You must provide a critique and score for each principle listed above.

Output ONLY a JSON object containing a single key "critiques_scores" which maps to a list of JSON objects. 
Each object in the list should correspond to one principle and contain the keys "principle", "critique", and "score".

Example format (if 2 principles were provided):
{{ 
  "critiques_scores": [
    {{
      "principle": "Principle 1 text...",
      "critique": "Critique relating to principle 1...",
      "score": 7
    }},
    {{
      "principle": "Principle 2 text...",
      "critique": "Critique relating to principle 2...",
      "score": 5
    }}
  ]
}}
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

    def _call_judge_with_retry(
        self, 
        judge_model: ModelInterface, 
        system_prompt: str, 
        user_prompt: str, 
        max_retries: int = 2,
        max_new_tokens: int = 150, # Increased for JSON
        temperature: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        """Calls the judge model, expects JSON output, and retries on parsing errors."""
        for attempt in range(max_retries + 1):
            try:
                response_raw = judge_model.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                # Sometimes the model might wrap JSON in ```json ... ```
                if response_raw.strip().startswith("```json"):
                    response_clean = response_raw.strip().split("```json")[1].split("```")[0].strip()
                else:
                    response_clean = response_raw.strip()
                    
                # Attempt to remove trailing commas before parsing
                response_clean = re.sub(r",(\s*[]}])", r"\1", response_clean)
                    
                parsed_json = json.loads(response_clean)
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"Attempt {attempt + 1}: Judge response parsing failed: {e}. Raw response: '{response_raw}'")
                if attempt == max_retries:
                    print("Max retries reached for judge call. Returning None.")
                    return None
            except Exception as e: # Catch other potential errors during generation/parsing
                 print(f"Attempt {attempt + 1}: An unexpected error occurred: {e}. Raw response: '{response_raw}'")
                 if attempt == max_retries:
                    print("Max retries reached due to unexpected error. Returning None.")
                    return None
        return None # Should not be reached, but added for safety

    def _perform_grm_evaluation(
        self,
        judge_model: ModelInterface,
        topic: str,
        arg1_text: str,
        arg2_text: str,
        num_principles: int,
        num_inference_rounds: int,
        device: str # Keep device for potential future GPU use in helpers
    ) -> Tuple[int, Optional[List[Dict[str, Any]]]]:
        """
        Performs the multi-round GRM-style evaluation for two arguments.

        Returns:
            - Overall winner (1 or 2)
            - Detailed log list for each round (or None if logging disabled/failed)
        """
        arg1_round_wins = 0
        arg2_round_wins = 0
        round_logs = []

        for round_num in range(num_inference_rounds):
            round_log = {"round": round_num + 1}
            
            # 1. Generate Principles
            principles_prompt = self.principles_prompt_template.format(num_principles=num_principles,topic=topic,arg1_response=arg1_text,arg2_response=arg2_text)

            principles_json = self._call_judge_with_retry(
                judge_model,
                system_prompt="You are an impartial debate judge generating evaluation principles.",
                user_prompt=principles_prompt,
                max_new_tokens=1000 # Allow more tokens for principles
            )

            if not principles_json or "principles" not in principles_json or not isinstance(principles_json["principles"], list):
                print(f"Round {round_num + 1}: Failed to get valid principles. Skipping round.")
                round_log["error"] = "Failed to generate principles"
                round_logs.append(round_log)
                continue
            
            principles = principles_json["principles"]
            round_log["principles"] = principles
            principles_list_str = "\n".join([f"- {p}" for p in principles])

            # 2. Critique and Score Argument 1
            critique1_prompt = self.critique_prompt_template.format(
                topic=topic,
                argument=arg1_text,
                principles_list_str=principles_list_str
            )
            critique1_json = self._call_judge_with_retry(
                judge_model,
                system_prompt="You are an impartial debate judge providing critique and scores based on principles.",
                user_prompt=critique1_prompt,
                max_new_tokens=1000
            )


            # 3. Critique and Score Argument 2
            critique2_prompt = self.critique_prompt_template.format(
                topic=topic,
                argument=arg2_text,
                principles_list_str=principles_list_str
            )
            critique2_json = self._call_judge_with_retry(
                judge_model,
                system_prompt="You are an impartial debate judge providing critique and scores based on principles.",
                user_prompt=critique2_prompt,
                max_new_tokens=1000
            )
            # Validate structure and extract critiques/scores
            critiques1 = critique1_json.get("critiques_scores", [])
            critiques2 = critique2_json.get("critiques_scores", [])

            # Basic validation (check if lists are received, more robust checks could be added)
            if not isinstance(critiques1, list) or not isinstance(critiques2, list):
                print(f"Round {round_num + 1}: critiques_scores is not a list. Skipping round.")
                round_log["error"] = "Invalid format for critiques_scores"
                round_logs.append(round_log)
                continue
            
            # Ensure each item has the required keys and score is int-like
            def validate_critiques(crit_list):
                total_score = 0
                valid = True
                for item in crit_list:
                    if not isinstance(item, dict) or not all(k in item for k in ["principle", "critique", "score"]):
                        valid = False
                        break
                    try:
                        total_score += int(item["score"])
                    except (ValueError, TypeError):
                        valid = False
                        break
                return valid, total_score

            valid1, score1_agg = validate_critiques(critiques1)
            valid2, score2_agg = validate_critiques(critiques2)

            if not valid1 or not valid2:
                print(f"Round {round_num + 1}: Invalid item format within critiques_scores list. Skipping round.")
                round_log["error"] = "Invalid item format in critiques_scores"
                round_logs.append(round_log)
                continue

            # Log the detailed list of critiques/scores
            round_log["arg1_critiques_scores"] = critiques1
            round_log["arg2_critiques_scores"] = critiques2
            round_log["arg1_aggregate_score"] = score1_agg
            round_log["arg2_aggregate_score"] = score2_agg
            
            # 4. Determine Round Winner based on aggregate scores
            if score1_agg > score2_agg:
                round_winner = 1
                arg1_round_wins += 1
            elif score2_agg > score1_agg:
                round_winner = 2
                arg2_round_wins += 1
            else: # Tie-breaking on aggregate score
                round_winner = random.choice([1, 2])
                if round_winner == 1:
                    arg1_round_wins += 1
                else:
                    arg2_round_wins += 1
            round_log["round_winner"] = f"Argument {round_winner}"
            round_logs.append(round_log)

        # Determine overall winner
        if arg1_round_wins > arg2_round_wins:
            overall_winner = 1
        elif arg2_round_wins > arg1_round_wins:
            overall_winner = 2
        else: # Tie in overall rounds
            overall_winner = random.choice([1, 2]) 

        return overall_winner, round_logs

    def _compute_train_rewards(
        self,
        input_prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        num_principles: int, # Added
        num_inference_rounds: int, # Added
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]: # Return type unchanged for train
        """Round-robin tournament scoring for training using GRM evaluation + format rewards."""
        num_completions = len(train_model_completions)
            
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Track wins/losses for each completion
        wins = torch.zeros(num_completions, device=device)
        losses = torch.zeros(num_completions, device=device)
        judge_model = all_models["judge_model"]
        topic = input_prompt.split('\nPosition:')[0].split("Debate Topic: ")[1]

        # Get debate scores using round-robin tournament with GRM evaluation
        for i in tqdm(range(num_completions), desc="Evaluating completions (Train)", leave=False):
            for j in range(i + 1, num_completions):
                response1_raw = train_model_completions[i]
                response2_raw = train_model_completions[j]
                response1 = self._extract_xml_answer(response1_raw)
                response2 = self._extract_xml_answer(response2_raw)
                
                # Perform GRM evaluation
                overall_winner, _ = self._perform_grm_evaluation( # Discard logs for training
                    judge_model=judge_model,
                    topic=topic,
                    arg1_text=response1,
                    arg2_text=response2,
                    num_principles=num_principles,
                    num_inference_rounds=num_inference_rounds,
                    device=device 
                )
                
                if overall_winner == 1:
                    wins[i] += 1
                    losses[j] += 1
                else: # overall_winner == 2
                    wins[j] += 1
                    losses[i] += 1

        # Calculate normalized scores (-1.5 to 1.5 range)
        total_matches = num_completions - 1 # number of matches per completion
        # Avoid division by zero if total_matches is 0 (num_completions=1 edge case handled above)
        win_rate = wins / total_matches if total_matches > 0 else torch.zeros_like(wins)
        loss_rate = losses / total_matches if total_matches > 0 else torch.zeros_like(losses)
        debate_scores = (win_rate - loss_rate) * 1.5 # Scale to desired range

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
        num_principles: int, # Added
        num_inference_rounds: int, # Added
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float], List[Dict[str, Any]]]: # Return type changed
        """Head-to-head debates against base model for testing using GRM + format rewards."""
        num_debates = len(train_model_completions)
        rewards_per_func = torch.zeros(num_debates, self.num_reward_functions, device=device)
        wins = 0
        detailed_logs = [] # Initialize log collector
        judge_model = all_models["judge_model"]
        
        # Get format rewards (calculated once upfront)
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
        
        for i in tqdm(range(num_debates), desc="Evaluating completions (Test)", leave=False):
            # Get trained model's response
            trained_response_raw = train_model_completions[i]
            trained_response = self._extract_xml_answer(trained_response_raw)
            
            # Get compare model's response
            compare_response_raw = compare_model_completions[i]
            compare_response = self._extract_xml_answer(compare_response_raw)     

            # Perform GRM evaluation
            overall_winner, round_logs = self._perform_grm_evaluation(
                 judge_model=judge_model,
                 topic=topic,
                 arg1_text=trained_response, # Trained model is Argument 1
                 arg2_text=compare_response, # Compare model is Argument 2
                 num_principles=num_principles,
                 num_inference_rounds=num_inference_rounds,
                 device=device
            )

            # Log details for this comparison
            log_entry = {
                "comparison_index": i,
                "topic": topic,
                "argument1_trained_raw": trained_response_raw,
                "argument2_compare_raw": compare_response_raw,
                "argument1_trained_extracted": trained_response,
                "argument2_compare_extracted": compare_response,
                "rounds": round_logs,
                "overall_winner": f"Argument {overall_winner}" 
            }
            detailed_logs.append(log_entry)
            
            # Assign debate score based on overall winner
            if overall_winner == 1: # Trained model (Argument 1) won
                score = 1.0
                rewards_per_func[i, 0] = score
                wins += 1
            else: # Compare model (Argument 2) won or tie went to compare
                score = 0.0 # Assign 0 for loss
                rewards_per_func[i, 0] = score


            # Add format rewards
            rewards_per_func[i, 1] = strict_format[i]
            rewards_per_func[i, 2] = soft_format[i]
            rewards_per_func[i, 3] = xml_count[i]

        win_rate = wins / num_debates if num_debates > 0 else 0.0
        # Calculate mean debate score (will be same as win_rate if score is 1 for win, 0 for loss)
        mean_debate_score = rewards_per_func[:, 0].mean().item() 
        
        metrics = {
            "win_rate": win_rate,
            # "reward": rewards_per_func.mean().item(), # This calculates mean over all reward functions
            "rewards/debate_score": mean_debate_score, # Report the mean debate score (win rate)
            "num_wins": wins,
            "num_debates": num_debates,
            "rewards/strict_format": strict_format.mean().item(),
            "rewards/soft_format": soft_format.mean().item(), 
            "rewards/xml_count": xml_count.mean().item(),
            "reward": rewards_per_func.sum(dim=1).mean().item() # Keep overall mean reward across all funcs
        }
        
        return rewards_per_func, metrics, detailed_logs # Return logs

    def compute_rewards(
        self,
        input_prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: Optional[List[str]] = None,
        num_principles: int = 3, # Added default
        num_inference_rounds: int = 3, # Added default
        device: str = "cuda",
        is_test: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float], Optional[List[Dict[str, Any]]]]: # Return type changed
        """Compute rewards - uses GRM evaluation for debate score."""
        detailed_logs = None # Initialize logs as None
        if is_test:
            if compare_model_completions is None:
                 raise ValueError("compare_model_completions must be provided when is_test=True")
            rewards_per_func, metrics, detailed_logs = self._compute_test_rewards(
                prompt=input_prompt, 
                all_models=all_models, 
                train_model_completions=train_model_completions, 
                compare_model_completions=compare_model_completions, 
                num_principles=num_principles,
                num_inference_rounds=num_inference_rounds,
                device=device
            )
        else:
            rewards_per_func, metrics = self._compute_train_rewards(
                input_prompt=input_prompt, 
                all_models=all_models, 
                train_model_completions=train_model_completions,
                num_principles=num_principles,
                num_inference_rounds=num_inference_rounds, 
                device=device
            )
            
        return rewards_per_func, metrics, detailed_logs # Return logs (will be None if not is_test)
            
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


