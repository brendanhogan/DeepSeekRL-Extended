"""
Abstract base class and implementations for reward computation in RL training.

"""

import re
import torch
import math
import numpy as np # Added numpy
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional

from clock_generator import TimeObj # Import TimeObj
from gui_generator import IMAGE_WIDTH, IMAGE_HEIGHT # For max distance calculation

# Maximum possible time difference on a 12-hour clock in seconds (6 hours)
MAX_DIFF_SECONDS = 6 * 3600 
MAX_POSSIBLE_DISTANCE_GUI = math.sqrt(IMAGE_WIDTH**2 + IMAGE_HEIGHT**2) # Diagonal of the image

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
    if name.lower() == "clock":
        return ClockEvaluator()
    elif name.lower() == "correlation":
        return CorrelationEvaluator()
    elif name.lower() == "gui":
        return GUIEvaluator()
    elif name.lower() == "captcha":
        return CaptchaEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}. Supported: 'clock', 'correlation', 'gui', 'captcha'")


class ClockEvaluator(RewardEvaluator):
    """
    Reward evaluator for the Analog Clock time prediction task.
    
    Implements reward functions for:
    - Time correctness (based on seconds difference, scaled from +3 to -3)
    - HH:MM:SS format correctness
    - Strict XML formatting (<reasoning>/<answer> tags)
    """
    
    def __init__(self, accuracy_tolerance_seconds: int = 60):
        self.num_reward_functions = 3 # Correctness, Time Format, XML Format
        self.accuracy_tolerance_seconds = accuracy_tolerance_seconds
        # Regex to extract HH:MM:SS, ensuring it's not part of a larger number
        self.time_extract_pattern = re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})\b")
        # Regex for the strict overall XML format
        self.strict_xml_pattern = re.compile(r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$", re.DOTALL)

    def _extract_time_string(self, text: str) -> str | None:
        """Extract HH:MM:SS time string from the <answer> tag."""
        try:
            # Isolate content within <answer> tags
            answer_content = text.split("<answer>")[-1].split("</answer>")[0].strip()
            # Match the HH:MM:SS pattern within the answer content
            match = self.time_extract_pattern.search(answer_content)
            if match:
                # Return the matched time string directly
                return match.group(0) # Return the exact match found
            return None
        except IndexError:
            return None # Tags not found or incorrect structure

    def _time_format_reward(self, extracted_times: List[str | None]) -> List[float]:
        """Reward for having the correct [HH:MM:SS] format extracted.
           Awards 0.5 if the format was successfully extracted, 0 otherwise.
        """
        # The extraction itself validates the format based on the regex
        return [0.5 if time_str is not None else 0.0 for time_str in extracted_times]

    def _correctness_reward(self, extracted_times: List[str | None], ground_truth_answers: List[str]) -> Tuple[List[float], List[float]]:
        """Reward based on time difference in seconds, scaled from +3 to -3.
           Returns a tuple: (list of reward scores, list of absolute errors in seconds)
        """
        rewards = []
        abs_errors = []
        max_reward = 3.0
        min_reward = -3.0

        for pred_time_str, true_time_str in zip(extracted_times, ground_truth_answers):

            true_time_obj = TimeObj.from_string(true_time_str)
            pred_time_obj = TimeObj.from_string(pred_time_str) if pred_time_str else None

            if pred_time_obj is None:
                # Prediction is invalid or couldn't be parsed
                rewards.append(min_reward)
                abs_errors.append(float(MAX_DIFF_SECONDS)) # Assign max error if format is wrong
            else:
                # Both times are valid, calculate difference
                diff_seconds, _ = true_time_obj.subtract(pred_time_obj)
                abs_errors.append(float(diff_seconds))
                
                # Scale reward linearly from +3 (0 diff) to -3 (MAX_DIFF_SECONDS diff)
                scaled_reward = max_reward - (max_reward - min_reward) * (diff_seconds / MAX_DIFF_SECONDS)
                rewards.append(scaled_reward)
                
        return rewards, abs_errors

    def _strict_xml_format_reward(self, completions: List[List[Dict[str, str]]]) -> List[float]:
        """Reward for strict <reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n format."""
        responses = [comp[0]['content'] for comp in completions]
        matches = [bool(self.strict_xml_pattern.match(r)) for r in responses]
        return [0.5 if m else 0.0 for m in matches]

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answers: List[str], # Expecting a list of ground truth time strings "[HH:MM:SS]"
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the clock task."""

        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Extract predicted time strings
        extracted_times = [self._extract_time_string(comp[0]['content']) for comp in completions]

        # Compute reward components
        correctness_scores, abs_error_seconds = self._correctness_reward(extracted_times, answers)
        time_format_scores = self._time_format_reward(extracted_times)
        xml_format_scores = self._strict_xml_format_reward(completions)

        all_scores = [
            correctness_scores,
            time_format_scores,
            xml_format_scores
        ]
        
        # Fill rewards tensor
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        abs_error_tensor = torch.tensor(abs_error_seconds, dtype=torch.float32, device=device)
        mean_abs_error = abs_error_tensor.mean().item()
        
        # Calculate accuracy (within tolerance)
        num_accurate = (abs_error_tensor <= self.accuracy_tolerance_seconds).sum().item()
        accuracy = num_accurate / num_completions if num_completions > 0 else 0.0
        
        mean_abs_error_minutes = mean_abs_error / 60.
        mean_abs_error_hours = mean_abs_error / 3600.

        metrics = {
            "rewards/correctness_reward_func": reward_per_func[0].item(),
            "rewards/time_format_reward_func": reward_per_func[1].item(), 
            "rewards/strict_xml_format_reward_func": reward_per_func[2].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(), # Total reward mean
            "metrics/mean_abs_error_seconds": mean_abs_error,
            "metrics/mean_abs_error_minutes": mean_abs_error_minutes,
            "metrics/mean_abs_error_hours": mean_abs_error_hours,
            "metrics/accuracy": accuracy # Accuracy within tolerance
        }
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        # Ensure reward_scores is a 1D tensor with expected length
        if reward_scores.ndim == 1 and len(reward_scores) == self.num_reward_functions:
             return {
                 'correctness': reward_scores[0].item(),
                 'time_format': reward_scores[1].item(),
                 'strict_xml_format': reward_scores[2].item(),
             }
        elif reward_scores.ndim == 2 and reward_scores.shape[1] == self.num_reward_functions:
            # If passed the whole batch tensor, return breakdown for the first element
            # Or consider averaging? Returning first for now.
            print("Warning: get_reward_breakdown received batch tensor, returning breakdown for first item.")
            return {
                 'correctness': reward_scores[0, 0].item(),
                 'time_format': reward_scores[0, 1].item(),
                 'strict_xml_format': reward_scores[0, 2].item(),
             }
        else:
             print(f"Warning: Unexpected shape for reward_scores in get_reward_breakdown: {reward_scores.shape}")
             # Return default/empty breakdown
             return {
                 'correctness': 0.0,
                 'time_format': 0.0,
                 'strict_xml_format': 0.0,
             }

# --- Correlation Scatter Plot Evaluator --- 

class CorrelationEvaluator(RewardEvaluator):
    """
    Reward evaluator for the Correlation Scatter Plot estimation task.
    
    Implements reward functions for:
    - Correlation correctness (based on absolute difference, scaled 0 to 1)
    - X.XX format correctness
    - Strict XML formatting (<reasoning>/<answer> tags)
    """
    
    def __init__(self):
        self.num_reward_functions = 3 # Correctness, Value Format, XML Format
        # Regex to extract X.XX format (0.00 to 1.00)
        self.correlation_extract_pattern = re.compile(r"\b([01]\.\d{2})\b") 
        # Regex for the strict overall XML format (same as clock task)
        self.strict_xml_pattern = re.compile(r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\\n</answer>\n$", re.DOTALL)

    def _extract_correlation_value(self, text: str) -> float | None:
        """Extract X.XX correlation value from the <answer> tag."""
        try:
            # Isolate content within <answer> tags
            answer_content = text.split("<answer>")[-1].split("</answer>")[0].strip()
            # Match the X.XX pattern within the answer content
            match = self.correlation_extract_pattern.search(answer_content)
            if match:
                # Return the matched correlation value as float
                value = float(match.group(1))
                # Ensure value is within [0.0, 1.0] range (due to regex \b1\.00\b is allowed)
                if 0.0 <= value <= 1.0:
                    return value
            return None
        except (IndexError, ValueError):
            return None # Tags not found, incorrect structure, or float conversion failed

    def _correlation_format_reward(self, extracted_values: List[float | None]) -> List[float]:
        """Reward for having the correct X.XX format extracted.
           Awards 0.5 if the format was successfully extracted, 0 otherwise.
        """
        # The extraction itself validates the format based on the regex and range check
        return [0.5 if value is not None else 0.0 for value in extracted_values]

    def _correctness_reward(self, extracted_values: List[float | None], ground_truth_answers: List[str]) -> Tuple[List[float], List[float]]:
        """Reward based on absolute difference, scaled linearly from +1 (0 diff) to 0 (1.0 diff).
           Returns a tuple: (list of reward scores, list of absolute errors)
        """
        rewards = []
        abs_errors = []
        max_reward = 1.0
        min_reward = 0.0
        max_possible_error = 1.0

        for pred_val, true_r_str in zip(extracted_values, ground_truth_answers):
            try:
                true_r = float(true_r_str) # Ground truth is already "X.XX"
            except ValueError:
                 # Should not happen if dataloader is correct
                 print(f"Warning: Could not parse ground truth R value: {true_r_str}")
                 rewards.append(min_reward)
                 abs_errors.append(max_possible_error)
                 continue

            if pred_val is None:
                # Prediction is invalid or couldn't be parsed
                rewards.append(min_reward)
                abs_errors.append(max_possible_error) # Assign max error if format is wrong
            else:
                # Both values are valid floats between 0 and 1
                diff = abs(true_r - pred_val)
                abs_errors.append(diff)
                
                # Scale reward linearly: Reward = MaxReward - (Diff / MaxError) * (MaxReward - MinReward)
                # Since MaxReward=1, MinReward=0, MaxError=1, this simplifies:
                scaled_reward = max_reward - diff 
                rewards.append(scaled_reward)
                
        return rewards, abs_errors

    def _strict_xml_format_reward(self, completions: List[List[Dict[str, str]]]) -> List[float]:
        """Reward for strict <reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n format."""
        responses = [comp[0]['content'] for comp in completions]
        matches = [bool(self.strict_xml_pattern.match(r)) for r in responses]
        return [0.5 if m else 0.0 for m in matches] # Award 0.5 for correct XML

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answers: List[str], # Expecting a list of ground truth R strings "X.XX"
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the correlation task."""

        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Extract predicted correlation values
        extracted_values = [self._extract_correlation_value(comp[0]['content']) for comp in completions]

        # Compute reward components
        correctness_scores, abs_error_values = self._correctness_reward(extracted_values, answers)
        correlation_format_scores = self._correlation_format_reward(extracted_values)
        xml_format_scores = self._strict_xml_format_reward(completions)

        all_scores = [
            correctness_scores, # Scaled 0-1
            correlation_format_scores, # 0 or 0.5
            xml_format_scores # 0 or 0.5
        ]
        
        # Fill rewards tensor
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        abs_error_tensor = torch.tensor(abs_error_values, dtype=torch.float32, device=device)
        mean_abs_error = abs_error_tensor.mean().item()
        
        # Total reward is sum of components (max possible is 1.0 + 0.5 + 0.5 = 2.0)
        total_reward_mean = rewards_per_func.sum(dim=1).mean().item()

        metrics = {
            "rewards/correctness_reward_func": reward_per_func[0].item(),
            "rewards/correlation_format_reward_func": reward_per_func[1].item(), 
            "rewards/strict_xml_format_reward_func": reward_per_func[2].item(),
            "reward": total_reward_mean, # Total reward mean
            "metrics/mean_abs_correlation_error": mean_abs_error,
        }
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        # Ensure reward_scores is a 1D tensor with expected length
        if reward_scores.ndim == 1 and len(reward_scores) == self.num_reward_functions:
             return {
                 'correctness': reward_scores[0].item(),
                 'correlation_format': reward_scores[1].item(),
                 'strict_xml_format': reward_scores[2].item(),
             }
        elif reward_scores.ndim == 2 and reward_scores.shape[1] == self.num_reward_functions:
            # Handle batch tensor case (return first item's breakdown)
            print("Warning: get_reward_breakdown received batch tensor, returning breakdown for first item.")
            return {
                 'correctness': reward_scores[0, 0].item(),
                 'correlation_format': reward_scores[0, 1].item(),
                 'strict_xml_format': reward_scores[0, 2].item(),
             }
        else:
             print(f"Warning: Unexpected shape for reward_scores in get_reward_breakdown: {reward_scores.shape}")
             # Return default/empty breakdown
             return {
                 'correctness': 0.0,
                 'correlation_format': 0.0,
                 'strict_xml_format': 0.0,
             }

# --- GUI Interaction Evaluator --- 

class GUIEvaluator(RewardEvaluator):
    """
    Reward evaluator for the GUI Interaction (click prediction) task.
    
    Implements reward functions for:
    - Strict XML formatting (<reasoning>/<answer>x,y</answer>)
    - Click Hit (whether the click is within the target bounding box)
    - Distance to Center (Euclidean distance from click to target center, scaled)
    """
    
    def __init__(self):
        self.num_reward_functions = 3 # XML Format, Click Hit, Distance to Center
        # Regex to extract "x,y" coordinates, allowing for spaces around comma
        self.coord_extract_pattern = re.compile(r"(\d+)\s*,\s*(\d+)")
        # Regex for the strict overall XML format, ensuring x,y in answer
        # This pattern is more specific for the answer part to guide the LLM.
        self.strict_xml_pattern = re.compile(r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n\d+\s*,\s*\d+\n</answer>\n$", re.DOTALL)
        # Define names of known circular elements
        self.circular_elements = {"window_close_button", "window_minimize_button", "window_maximize_button"}

    def _extract_coordinates(self, text: str) -> Optional[Tuple[int, int]]:
        """Extract x,y coordinates from the <answer> tag."""
        try:
            answer_content = text.split("<answer>")[-1].split("</answer>")[0].strip()
            match = self.coord_extract_pattern.search(answer_content)
            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                # Basic check for coordinates being within typical image bounds (e.g., 0-1024, can be refined based on actual image size if needed)
                # For now, GUIGenerator uses 224x224. We can add stricter checks if x/y are way off.
                if 0 <= x <= IMAGE_WIDTH*2 and 0 <= y <= IMAGE_HEIGHT*2: # Allow some leeway beyond 224 for robustness
                    return x, y
            return None
        except (IndexError, ValueError):
            return None # Tags not found, incorrect structure, or int conversion failed

    def _is_click_in_bbox(self, click_xy: Optional[Tuple[int, int]], target_bbox: Tuple[int, int, int, int], target_name: Optional[str] = None) -> bool:
        """Check if the click (x,y) is within the target area (bbox or circle radius)."""
        if click_xy is None:
            return False
        x, y = click_xy
        x_min, y_min, x_max, y_max = target_bbox

        # Check for known circular elements
        if target_name and target_name in self.circular_elements:
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            radius = (x_max - x_min) / 2 # Assume bbox tightly bounds the circle
            distance_from_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            return distance_from_center <= radius
        else:
            # Default rectangular bounding box check
            return x_min <= x <= x_max and y_min <= y <= y_max

    def _strict_xml_format_reward(self, completions: List[List[Dict[str, str]]]) -> List[float]:
        """Reward for strict <reasoning>...</reasoning><answer>x,y</answer> format."""
        responses = [comp[0]['content'] for comp in completions]
        # Check overall structure and if coordinates can be extracted (implies x,y format in answer is somewhat met)
        rewards = []
        for r in responses:
            xml_match = bool(self.strict_xml_pattern.match(r))
            coords_extracted = self._extract_coordinates(r) is not None
            if xml_match and coords_extracted:
                rewards.append(0.5)
            elif coords_extracted and not xml_match: # Has x,y but not full XML structure
                rewards.append(0.1) # Small partial credit for at least getting coordinates
            else:
                rewards.append(0.0)
        return rewards

    def _click_hit_reward(self, extracted_coords: List[Optional[Tuple[int, int]]], 
                          target_bboxes: List[Tuple[int, int, int, int]],
                          target_names: List[str]) -> List[float]:
        """Reward +3 if click is inside target area (bbox or circle), 0 otherwise."""
        rewards = []
        for click_xy, bbox, name in zip(extracted_coords, target_bboxes, target_names):
            # Pass target_name to the check function
            if self._is_click_in_bbox(click_xy, bbox, name):
                rewards.append(3.0)
            else:
                rewards.append(0.0)
        return rewards

    def _distance_to_center_reward(self, extracted_coords: List[Optional[Tuple[int, int]]], 
                                   target_centers: List[Tuple[int,int]], 
                                   target_bboxes: List[Tuple[int,int,int,int]]) -> Tuple[List[float], List[float]]:
        """Scaled reward based on Euclidean distance to target center (+2 to -2).
           Returns rewards and the raw distance errors.
        """
        rewards = []
        distance_errors = [] # Store raw distance errors for metrics
        
        max_reward = 2.0
        min_reward = -2.0 # Furthest possible click, or unparseable

        for click_xy, target_center_xy, target_bbox in zip(extracted_coords, target_centers, target_bboxes):
            if click_xy is None: # Coordinate couldn't be parsed
                rewards.append(min_reward)
                distance_errors.append(MAX_POSSIBLE_DISTANCE_GUI) # Max possible error
                continue

            pred_x, pred_y = click_xy
            true_center_x, true_center_y = target_center_xy
            
            distance = math.sqrt((pred_x - true_center_x)**2 + (pred_y - true_center_y)**2)
            distance_errors.append(distance)
            
            # Scale reward: +2 for 0 distance, down to -2 for MAX_POSSIBLE_DISTANCE_GUI
            # Reward = MaxReward - (Distance / MaxDistance) * (MaxReward - MinReward)
            # Ensure distance doesn't exceed MAX_POSSIBLE_DISTANCE_GUI for scaling
            clamped_distance = min(distance, MAX_POSSIBLE_DISTANCE_GUI)
            scaled_reward = max_reward - (clamped_distance / MAX_POSSIBLE_DISTANCE_GUI) * (max_reward - min_reward)
            rewards.append(scaled_reward)
            
        return rewards, distance_errors

    def compute_rewards(
        self,
        prompts: Optional[List[List[Dict[str, str]]]], # Prompts may not be directly used here but are part of API
        completions: List[List[Dict[str, str]]],
        answers: List[Dict[str, Any]], # List of target_info dicts from GUIDataLoader
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the GUI click task."""
        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        extracted_coords_list = [self._extract_coordinates(comp[0]['content']) for comp in completions]
        target_bboxes_list = [ans['bounding_box'] for ans in answers]
        target_centers_list = [(ans['center_x'], ans['center_y']) for ans in answers]
        target_names_list = [ans['name'] for ans in answers] # Get target names

        # 1. XML Format Reward
        xml_format_scores = self._strict_xml_format_reward(completions)
        
        # 2. Click Hit Reward (pass target names)
        click_hit_scores = self._click_hit_reward(extracted_coords_list, target_bboxes_list, target_names_list)
        
        # 3. Distance to Center Reward
        dist_rewards_scores, raw_distance_errors = self._distance_to_center_reward(extracted_coords_list, target_centers_list, target_bboxes_list)

        all_component_scores = [
            xml_format_scores,
            click_hit_scores,
            dist_rewards_scores
        ]
        
        for i, scores_component in enumerate(all_component_scores):
            rewards_per_func[:, i] = torch.tensor(scores_component, dtype=torch.float32, device=device)
        
        # --- Metrics --- 
        # Mean for each reward component
        mean_rewards_per_component = rewards_per_func.mean(dim=0)
        
        # Click Hit Rate (Accuracy)
        num_hits = sum(1 for score in click_hit_scores if score > 0) # count non-zero scores
        click_hit_rate = num_hits / num_completions if num_completions > 0 else 0.0
        
        # Mean Distance Error
        distance_errors_tensor = torch.tensor(raw_distance_errors, dtype=torch.float32, device=device)
        mean_dist_error = distance_errors_tensor.mean().item()
        
        # Total reward mean
        total_reward_mean = rewards_per_func.sum(dim=1).mean().item()

        metrics = {
            "rewards/xml_format_reward": mean_rewards_per_component[0].item(),
            "rewards/click_hit_reward": mean_rewards_per_component[1].item(),
            "rewards/distance_to_center_reward": mean_rewards_per_component[2].item(),
            "reward": total_reward_mean, # Overall mean reward
            "metrics/click_hit_rate": click_hit_rate,
            "metrics/mean_distance_to_center_error": mean_dist_error
        }
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        if reward_scores.ndim == 1 and len(reward_scores) == self.num_reward_functions:
            return {
                'xml_format': reward_scores[0].item(),
                'click_hit': reward_scores[1].item(),
                'distance_to_center': reward_scores[2].item(),
            }
        elif reward_scores.ndim == 2 and reward_scores.shape[1] == self.num_reward_functions:
            # For batch tensor, return for the first item (or mean if preferred)
            return {
                'xml_format': reward_scores[0, 0].item(),
                'click_hit': reward_scores[0, 1].item(),
                'distance_to_center': reward_scores[0, 2].item(),
            }
        else:
            # print(f"Warning: Unexpected shape for reward_scores in GUIEvaluator.get_reward_breakdown: {reward_scores.shape}")
            return {'xml_format': 0.0, 'click_hit': 0.0, 'distance_to_center': 0.0}

# --- CAPTCHA Evaluator ---
from collections import Counter as CollectionCounter # To avoid conflict with local Counter

# Constants for CaptchaEvaluator rewards
# CAPTCHA_REWARD_PER_TP = 1.5  # Reward for each correctly clicked square
# CAPTCHA_PENALTY_PER_FP = -1.0 # Penalty for each incorrectly clicked square
# CAPTCHA_PENALTY_PER_FN = -0.75 # Penalty for each missed target square
CAPTCHA_XML_FORMAT_REWARD = 0.5 # Bonus for correct XML structure (kept as a metric)
# CAPTCHA_MAX_REWARD_NORMALIZATION = 5.0 # Target max reward for normalization - F1 is already 0-1
# CAPTCHA_MIN_REWARD_NORMALIZATION = -5.0 # Target min reward for normalization

class CaptchaEvaluator(RewardEvaluator):
    """
    Reward evaluator for the CAPTCHA square selection task.
    Rewards based on F1 score of identifying target squares using `click_screen(x,y)` calls.
    
    Note: A grid square should only be considered as containing the target object 
    if the target object occupies at least 20% of the square's area.
    This threshold should be applied during dataset generation in captcha_generator.py.
    """
    def __init__(self):
        self.click_call_pattern = re.compile(r"click_screen\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
        self.strict_xml_pattern = re.compile(r"^<reasoning>.*?<answer>.*?</answer>.*?$", re.DOTALL | re.IGNORECASE)
        self.answer_tag_content_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
        self.num_reward_functions = 1 # Solely F1 score now for the main reward tensor
        # Note: target_squares_boolean should be based on 20% object area threshold

    def _extract_click_calls(self, completion_text: str) -> List[Tuple[int, int]]:
        unique_clicks = set()
        try:
            answer_match = self.answer_tag_content_pattern.search(completion_text)
            if not answer_match:
                return [] 
            answer_content = answer_match.group(1)
            matches = self.click_call_pattern.findall(answer_content)
            for match in matches:
                try:
                    x = int(match[0])
                    y = int(match[1])
                    if 0 <= x < 224 and 0 <= y < 224:
                        unique_clicks.add((x, y))
                except ValueError:
                    continue 
        except Exception:
            pass 
        return list(unique_clicks)

    def _classify_clicks_and_get_stats(self, 
                                       predicted_clicks: List[Tuple[int, int]], 
                                       target_squares_boolean: List[bool],
                                       target_square_coords_final: List[List[float]] # Currently unused but kept for API consistency
                                       ) -> Dict[str, any]: # 'any' because f1 can be float
        num_total_squares = len(target_squares_boolean)
        clicked_ground_truth_indices = [False] * num_total_squares
        true_positives = 0
        false_positives = 0
        true_target_indices = {i for i, is_true in enumerate(target_squares_boolean) if is_true}
        num_actual_targets = len(true_target_indices)
        predicted_clicks_on_square_indices = []

        from captcha_generator import SQUARE_CROP_DIM, GRID_SIZE, CELL_DIM, BANNER_ABS_HEIGHT, FINAL_DIM, PADDING_SIZE
        scale_x = FINAL_DIM / (SQUARE_CROP_DIM + 2 * PADDING_SIZE)
        scale_y = FINAL_DIM / (BANNER_ABS_HEIGHT + SQUARE_CROP_DIM + 2 * PADDING_SIZE)
        grid_origin_x_on_padded = PADDING_SIZE
        grid_origin_y_on_padded = PADDING_SIZE + BANNER_ABS_HEIGHT
        final_cell_dim_x = (CELL_DIM * scale_x)
        final_cell_dim_y = (CELL_DIM * scale_y) 
        final_grid_start_x = grid_origin_x_on_padded * scale_x
        final_grid_start_y = grid_origin_y_on_padded * scale_y

        for pred_x, pred_y in predicted_clicks:
            found_match_in_any_square = False
            clicked_square_idx = -1
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    cell_idx = r * GRID_SIZE + c
                    sq_x1 = final_grid_start_x + c * final_cell_dim_x
                    sq_y1 = final_grid_start_y + r * final_cell_dim_y
                    sq_x2 = sq_x1 + final_cell_dim_x
                    sq_y2 = sq_y1 + final_cell_dim_y
                    if sq_x1 <= pred_x < sq_x2 and sq_y1 <= pred_y < sq_y2:
                        clicked_square_idx = cell_idx
                        found_match_in_any_square = True
                        break
                if found_match_in_any_square: break
            if found_match_in_any_square:
                predicted_clicks_on_square_indices.append(clicked_square_idx)
            else:
                false_positives += 1 
        
        click_counts_per_square = CollectionCounter(predicted_clicks_on_square_indices)
        for square_idx, num_clicks_in_square in click_counts_per_square.items():
            if square_idx in true_target_indices:
                true_positives += 1 
                clicked_ground_truth_indices[square_idx] = True 
                if num_clicks_in_square > 1:
                    false_positives += (num_clicks_in_square - 1) 
            else: 
                false_positives += num_clicks_in_square 

        false_negatives = num_actual_targets - sum(clicked_ground_truth_indices)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives,
            "num_actual_targets": num_actual_targets,
            "num_predicted_clicks": len(predicted_clicks),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    def compute_rewards(
        self,
        prompts: Optional[List[List[Dict[str, str]]]], 
        completions: List[List[Dict[str, str]]],
        answers: List[Dict[str, Any]], 
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        all_f1_scores = []
        all_tp = []
        all_fp = []
        all_fn = []
        all_num_targets = []
        all_num_predicted_clicks = []
        all_precisions = []
        all_recalls = []
        xml_format_scores_list = []

        for i in range(num_completions):
            completion_text = completions[i][0]['content']
            answer_data = answers[i]

            xml_format_score = CAPTCHA_XML_FORMAT_REWARD if self.strict_xml_pattern.search(completion_text) else 0.0
            xml_format_scores_list.append(xml_format_score)

            predicted_clicks = self._extract_click_calls(completion_text)
            target_squares_boolean = answer_data["target_squares_boolean"]
            stats = self._classify_clicks_and_get_stats(predicted_clicks, target_squares_boolean, []) 
            
            f1 = stats["f1_score"]
            rewards_per_func[i, 0] = f1 # F1 score is the direct reward signal
            
            all_f1_scores.append(f1)
            all_tp.append(stats["tp"])
            all_fp.append(stats["fp"])
            all_fn.append(stats["fn"])
            all_num_targets.append(stats["num_actual_targets"])
            all_num_predicted_clicks.append(stats["num_predicted_clicks"])
            all_precisions.append(stats["precision"])
            all_recalls.append(stats["recall"])

        metrics = {
            "rewards/f1_score_reward": torch.tensor(all_f1_scores, dtype=torch.float32, device=device).mean().item() if all_f1_scores else 0.0,
            "metrics/xml_format_score_avg": np.mean(xml_format_scores_list) if xml_format_scores_list else 0.0,
            "reward": rewards_per_func.sum(dim=1).mean().item(), # This will be mean F1 score
            "metrics/mean_true_positives": np.mean(all_tp) if all_tp else 0,
            "metrics/mean_false_positives": np.mean(all_fp) if all_fp else 0,
            "metrics/mean_false_negatives": np.mean(all_fn) if all_fn else 0,
            "metrics/precision": np.mean(all_precisions) if all_precisions else 0,
            "metrics/recall_percent_correct_squares": np.mean(all_recalls) if all_recalls else 0,
            "metrics/f1_score": np.mean(all_f1_scores) if all_f1_scores else 0, # Overall F1 for the batch
            "metrics/avg_predicted_clicks": np.mean(all_num_predicted_clicks) if all_num_predicted_clicks else 0,
        }
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        if reward_scores.ndim == 1 and len(reward_scores) == self.num_reward_functions: # Should be 1
            return {
                'f1_score': reward_scores[0].item(),
                # Add other components here if num_reward_functions > 1 in future
            }
        elif reward_scores.ndim == 2 and reward_scores.shape[1] == self.num_reward_functions:
            return {
                'f1_score': reward_scores[0, 0].item(),
            }
        else:
            return {'f1_score': 0.0}
