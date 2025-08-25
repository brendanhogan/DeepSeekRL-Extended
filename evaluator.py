"""
Abstract base class and implementations for reward computation in RL training.

"""
import re
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from contextlib import contextmanager
import signal
import pdb
import openai
import json
import multiprocessing
import traceback
import sys
import reasoning_gym
from multiprocessing import Process, Queue, TimeoutError as MPTimeoutError
def check_llm_answer(answer: str, ground_truth: str, evaluation_prompt: str="Determine if the response contains the ground truth number as its final answer.",ground_truth_number_prompt: str="Ground Truth Number:") -> bool:
    """Check if the answer is a valid answer to the question.
    Args:
        answer: The response from the model
        ground_truth: The ground truth answer
        evaluation_prompt: The prompt to use for evaluation
        ground_truth_number_prompt: The prompt to use for the ground truth number
    """

    client = openai.OpenAI(api_key='-', base_url="http://slurm-h100-206-119:8000/v1")
    model = client.models.list().data[0].id
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": evaluation_prompt},
        {"role": "user", "content": "Response: " + answer + "\n" + f"{ground_truth_number_prompt} {ground_truth}"}
    ],
    extra_body={"guided_choice": ["True", "False"]},
)
    return completion.choices[0].message.content == "True"
    response = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
            "role": "system",
            "content": [
                {
                "type": "input_text",
                "text": f"Determine if the response contains the ground truth number as its final answer. \n"
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "input_text",
                "text": "Response: " + answer + "\n" + f"Ground Truth Number: {ground_truth}"
                }
            ]
            }
        ],
        text={
            "format": {
            "type": "json_schema",
            "name": "final_answer_indicator",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                "is_ground_truth_in_response": {
                    "type": "boolean",
                    "description": "Indicator of whether the ground truth number is in the response."
                }
                },
                "required": [
                "is_ground_truth_in_response"
                ],
                "additionalProperties": False
            }
            }
        },
        think={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        store=True
    )
    response_json = json.loads(response.output_text)
    if "is_ground_truth_in_response" not in response_json:
        print("No is_ground_truth_in_response in response")
        return False
    return response_json["is_ground_truth_in_response"]

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds!")
    
    # This will only work on UNIX systems
    try:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        yield
    except ValueError: # signal only works in main thread
        # If not in the main thread, we can't set a signal handler.
        # In this case, we'll just run without a timeout.
        yield
    finally:
        try:
            signal.alarm(0)
        except ValueError:
            pass

def _execute_code_in_subprocess(code: str, test_cases: List[str], timeout: int, result_queue: Queue):
    """
    Execute code and test cases in a subprocess.
    
    This function runs in a separate process to isolate any crashes
    (including segmentation faults) from the parent process.
    
    Args:
        code: Python code to execute
        test_cases: List of test case strings to run
        timeout: Timeout in seconds for each execution
        result_queue: Queue to return results to parent process
    """
    try:
        # Create isolated execution environment
        test_globals = {}
        
        # Execute the generated code with timeout
        try:
            with time_limit(timeout):
                exec(code, test_globals)
        except TimeoutException:
            result_queue.put((False, 0, len(test_cases), "Code execution timeout"))
            return
        except Exception as e:
            result_queue.put((False, 0, len(test_cases), f"Code execution error: {str(e)}"))
            return
        
        if not test_cases:
            # If no test cases, just check if code runs successfully
            result_queue.put((True, 1, 1, "No test cases - code executed successfully"))
            return
        
        # Run each test case
        passed_count = 0
        total_count = len(test_cases)
        
        for i, test_case in enumerate(test_cases):
            try:
                with time_limit(timeout):
                    exec(test_case, test_globals)
                    passed_count += 1
            except TimeoutException:
                continue  # Test case timed out
            except Exception:
                continue  # Test case failed
        
        overall_pass = passed_count == total_count
        result_queue.put((overall_pass, passed_count, total_count, "Tests completed"))
        
    except Exception as e:
        # Catch any unexpected errors in the subprocess
        error_msg = f"Subprocess error: {str(e)}\n{traceback.format_exc()}"
        result_queue.put((False, 0, len(test_cases) if test_cases else 0, error_msg))

def safe_execute_code_with_tests(code: str, test_cases: List[str], timeout: int = 10) -> Tuple[bool, int, int]:
    """
    Safely execute code and test cases using subprocess isolation.
    
    This function protects the parent process from crashes (including segfaults)
    by running the code in a separate subprocess.
    
    Args:
        code: Python code to execute
        test_cases: List of test case strings to run
        timeout: Timeout in seconds for execution
        
    Returns:
        Tuple of (overall_pass, passed_count, total_count)
    """
    if not code.strip():
        return False, 0, len(test_cases) if test_cases else 0
    
    # Create a queue for inter-process communication
    result_queue = Queue()
    
    # Create and start subprocess
    process = Process(
        target=_execute_code_in_subprocess,
        args=(code, test_cases, timeout, result_queue)
    )
    
    try:
        process.start()
        
        # Wait for result with timeout (slightly longer than code timeout to allow cleanup)
        process_timeout = timeout * 2 + 5  # Give extra time for process overhead
        process.join(timeout=process_timeout)
        
        if process.is_alive():
            # Process is still running, terminate it
            process.terminate()
            process.join(timeout=5)  # Give it a chance to terminate gracefully
            
            if process.is_alive():
                # Force kill if it doesn't terminate
                process.kill()
                process.join()
            
            return False, 0, len(test_cases) if test_cases else 0
        
        # Check if we got a result
        if not result_queue.empty():
            overall_pass, passed_count, total_count, message = result_queue.get_nowait()
            return overall_pass, passed_count, total_count
        else:
            # Process ended without putting result (probably crashed)
            return False, 0, len(test_cases) if test_cases else 0
            
    except Exception as e:
        # Error in process management
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        return False, 0, len(test_cases) if test_cases else 0
    
    finally:
        # Ensure process is cleaned up
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()

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
        entry: dict | None = None
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
    if name.lower() == "gsm8k":
        return GSM8kEvaluator()
    elif name.lower() == "mbpp":
        return MBPPEvaluator()
    elif name.lower() == "leetcode":
        return MBPPEvaluator()
    elif name.lower() == "math500":
        return GSM8kEvaluator()
    elif "reasoning_gym" in name.lower():
        return thinkGymEvaluator(name.split(".")[0])
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")

class MBPPEvaluator(RewardEvaluator):
    """
    Reward evaluator for the MBPP (Mostly Basic Python Problems) dataset.
    
    Implements reward functions for:
    - Code correctness (passes test cases)
    - Syntax correctness
    - Code execution without errors
    - Format compliance
    - Code complexity/quality
    """
    def __init__(self, timeout: int = 5):
        self.num_reward_functions = 5
        self.timeout = timeout
    
    def _extract_code(self, text: str) -> str:
        """Extract code from text using patterns similar to mbpp.py"""
        import re
        
        # Try different code block patterns
        patterns = [
            r'```python\s*\n(.*?)\n```',  # ```python ... ```
            r'```\s*\n(.*?)\n```',       # ``` ... ```
            r'<code>(.*?)</code>',        # <code> ... </code>
            r'def\s+\w+\([^)]*\):.*?(?=\n\n|\n#|\nclass|\ndef|\Z)',  # Function definitions
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                # Return the first (usually longest) match
                code = matches[0].strip()
                if code and len(code) > 10:  # Basic sanity check
                    return code
        
        # If no explicit code blocks found, try to extract function definitions
        lines = text.split('\n')
        code_lines = []
        in_function = False
        
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                in_function = True
                code_lines.append(line)
            elif in_function:
                if line.strip() == '' or line.startswith((' ', '\t')):
                    code_lines.append(line)
                else:
                    # End of function
                    break
        
        if code_lines:
            return '\n'.join(code_lines)
        
        return ""
        
    def _check_syntax(self, code: str, function_signature: str) -> bool:
        """Check if the code is syntactically correct."""
        if not code.strip():
            return False
        try:
            import ast
            ast.parse(code)
            # Check if the function signature is in the code
            function_name = function_signature.split("(")[0]
            if function_name not in code:
                return False
            return True
        except SyntaxError:
            return False
        except Exception:
            return False
    
    def _correctness_reward(self, prompts, completions, answer) -> List[float]:
        """Reward for code that passes test cases."""
        responses = [completion[0]['content'] for completion in completions]
        rewards = []
        
        for response in responses:
            code = self._extract_code(response)
            if not code:
                rewards.append(0.0)
                continue
            
            test_cases = answer
            
            overall_pass, passed_count, total_count = safe_execute_code_with_tests(code, test_cases[0], self.timeout)
            
            if total_count == 0:
                # If no test cases, just check if code extracts and runs
                overall_pass_no_tests, _, _ = safe_execute_code_with_tests(code, [], self.timeout)
                if overall_pass_no_tests:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                # Reward based on test case pass rate
                if overall_pass:
                    rewards.append(2.0)
                else:
                    rewards.append(2.0 * (passed_count / total_count))
        
        return rewards

    def _syntax_reward(self, completions, function_signature) -> List[float]:
        """Reward for syntactically correct code."""
        responses = [completion[0]['content'] for completion in completions]
        rewards = []
        
        for response in responses:
            code = self._extract_code(response)
            if self._check_syntax(code, function_signature):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
                
        return rewards

    def _execution_reward(self, completions) -> List[float]:
        """Reward for code that executes without errors."""
        responses = [completion[0]['content'] for completion in completions]
        rewards = []
        
        for response in responses:
            code = self._extract_code(response)
            if not code:
                rewards.append(0.0)
                continue
                
            # Use safe subprocess execution to protect from crashes
            overall_pass, _, _ = safe_execute_code_with_tests(code, [], self.timeout)
            if overall_pass:
                rewards.append(0.5)
            else:
                rewards.append(0.0)
                
        return rewards

    def _format_reward(self, completions) -> List[float]:
        """Reward for properly formatted response with think tags and python code blocks."""
        responses = [completion[0]['content'] for completion in completions]
        rewards = []
        
        for response in responses:
            reward = 0.0
            
            # Check for think tags
            if '<think>' in response and '</think>' in response:
                reward += 1.0
            
            # Check for python code blocks
            if '```python' in response and '```' in response:
                reward += 1.0
            
            rewards.append(reward)
                
        return rewards

    def _code_quality_reward(self, completions) -> List[float]:
        """Reward for code quality (length, complexity, etc.)."""
        responses = [completion[0]['content'] for completion in completions]
        rewards = []
        
        for response in responses:
            code = self._extract_code(response)
            if not code:
                rewards.append(0.0)
                continue
                
            lines = len([line for line in code.split('\n') if line.strip()])
            
            # Reward based on reasonable code length (not too short, not too long)
            if 3 <= lines <= 50:
                rewards.append(0)
            elif 1 <= lines <= 100:
                rewards.append(0.2)
            else:
                rewards.append(0.0)
                
        return rewards

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        function_signature: str,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the given completions."""

        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)

        # Compute all reward functions
        all_scores = [
            self._correctness_reward(prompts, completions, answer),
            self._syntax_reward(completions, function_signature),
            self._execution_reward(completions),
            self._format_reward(completions),
            self._code_quality_reward(completions)
        ]
        
        # Fill rewards tensor
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        
        # Calculate accuracy (perfect correctness score)
        correctness_scores = rewards_per_func[:, 0]  # First reward function is correctness
        num_perfect = (correctness_scores == 2.0).sum().item()
        accuracy = num_perfect / num_completions if num_completions > 0 else 0.0
        
        metrics = {
            "rewards/correctness_reward_func": reward_per_func[0].item(),
            "rewards/syntax_reward_func": reward_per_func[1].item(), 
            "rewards/execution_reward_func": reward_per_func[2].item(),
            "rewards/format_reward_func": reward_per_func[3].item(),
            "rewards/quality_reward_func": reward_per_func[4].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "accuracy": accuracy
        }
        
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        return {
            'correctness': reward_scores[0].item(),
            'syntax': reward_scores[1].item(),
            'execution': reward_scores[2].item(),
            'format': reward_scores[3].item(),
            'quality': reward_scores[4].item()
        }

class thinkGymEvaluator(RewardEvaluator):
    """
    Reward evaluator for the think Gym dataset.
    """
    def __init__(self, dataset_name: str):
        self.num_reward_functions = 4
        self.dataset_name = dataset_name
        self.dataset = reasoning_gym.create_dataset(self.dataset_name, size=10000,seed=42)
    def _extract_answer(self, text: str) -> str:
        """Extract answer from text."""
        if "<answer>" in text:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            ### remove all newlines
            answer = answer.replace("\n", "")
            return answer.strip()
        else:
            return ""
    def _correctness_reward(self, prompts, completions, answer, entry: dict | None = None) -> List[float]:
        """Reward for correct answer."""
        responses = [self._extract_answer(completion[0]['content']) for completion in completions]
        rewards = []
        for r, a in zip(responses, answer):
            if r == "":
                rewards.append(0.0)
                continue
            if entry is not None:
                rewards.append(self.dataset.score_answer(answer=r,entry=entry)*2)
                continue
            else:
                print("No entry provided")
            if check_llm_answer(r[-4000:], a,evaluation_prompt="Determine if the response contains the ground truth answer in its final answer.",ground_truth_number_prompt="Ground Truth Answer:"):
                rewards.append(2.0)
            else:
                rewards.append(0.0)
        return rewards
    def _format_reward(self, completions) -> List[float]:
        """Reward for relaxed XML format."""
        pattern = r".*</think>\s*<answer>.*?</answer>\s*\Z"
        responses = [completion[0]["content"] for completion in completions]
        matches = [bool(re.match(pattern, r)) for r in responses]
        return [0.5 if m else 0.0 for m in matches]
    
    def _word_count_reward(self, completions) -> List[float]:
        """Reward for word count."""
        responses = [completion[0]["content"] for completion in completions]
        lengths = [len(r.split()) for r in responses]
        return [0.5 if l > 200 and l < 400 else 0.0 for l in lengths]

    def _xml_count_reward(self, completions) -> List[float]:
        """Reward for XML tag counting."""
        def count_xml(text: str) -> float:
            count = 0.0
            # if text.count("<think>\n") == 1: count += 0.5
            if text.count("\n</think>\n") == 1: count += 0.5
            if text.count("\n<answer>\n") == 1:
                count += 0.125
                count -= len(text.split("\n</answer>\n")[-1])*0.001
            if text.count("\n</answer>") == 1:
                count += 0.125
                count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
            return count
        responses = [completion[0]["content"] for completion in completions]
        return [count_xml(r) for r in responses]
        
    def compute_rewards(self, prompts, completions, answer, device, entry: dict | None = None):
        """Compute all rewards for the given completions."""
        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        all_scores = [
            self._correctness_reward(prompts, completions, answer, entry),
            self._format_reward(completions),
            self._word_count_reward(completions),
            self._xml_count_reward(completions)
        ]
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        reward_per_func = rewards_per_func.mean(0)
        correctness_scores = rewards_per_func[:, 0]  # First reward function is correctness
        num_perfect = (correctness_scores == 2.0).sum().item()
        accuracy = num_perfect / num_completions if num_completions > 0 else 0.0
        metrics={
            "rewards/correctness_reward_func": reward_per_func[0].item(),
            "rewards/format_reward_func": reward_per_func[1].item(),
            "rewards/word_count_reward_func": reward_per_func[2].item(),
            "rewards/xml_count_reward_func": reward_per_func[3].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "correctness": correctness_scores.tolist(),
            "accuracy": accuracy
        }
        return rewards_per_func, metrics
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        return {
            'correctness': reward_scores[0].item(),
            'format': reward_scores[1].item(),
            'word_count': reward_scores[2].item(),
            'xml_count': reward_scores[3].item()
        }

class GSM8kEvaluator(RewardEvaluator):
    """
    Reward evaluator for the GSM8K math problem dataset.
    
    Implements reward functions for:
    - Answer correctness
    - Integer format validation
    - XML formatting (strict and soft)
    - XML tag counting
    """
    
    def __init__(self):
        self.num_reward_functions = 5
    
    def _extract_xml_answer(self, text: str) -> str:
        """Extract answer from XML tags."""
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    
    def _correctness_reward(self, prompts, completions, answer) -> List[float]:
        """Reward for correct answer."""
        responses = [self._extract_xml_answer(completion[0]['content']) for completion in completions]
        #extracted = [self._extract_xml_answer(r) for r in responses]
        rewards = []
        for r, a in zip(responses, answer):
            if check_llm_answer(r[-4000:], a):
                rewards.append(2.0)
            else:
                rewards.append(0.0)
        return rewards

    def _int_format_reward(self, completions) -> List[float]:
        """Reward for integer format."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted]

    def _strict_format_reward(self, completions) -> List[float]:
        """Reward for strict XML format."""
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [bool(re.match(pattern, r)) for r in responses]
        return [0.5 if m else 0.0 for m in matches]

    def _soft_format_reward(self, completions) -> List[float]:
        """Reward for relaxed XML format."""
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [bool(re.match(pattern, r)) for r in responses]
        return [0.5 if m else 0.0 for m in matches]

    def _xml_count_reward(self, completions) -> List[float]:
        """Reward for XML tag counting."""
        def count_xml(text: str) -> float:
            count = 0.0
            if text.count("<think>\n") == 1: count += 0.125
            if text.count("\n</think>\n") == 1: count += 0.125
            if text.count("\n<answer>\n") == 1:
                count += 0.125
                count -= len(text.split("\n</answer>\n")[-1])*0.001
            if text.count("\n</answer>") == 1:
                count += 0.125
                count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
            return count
            
        responses = [completion[0]["content"] for completion in completions]
        return [count_xml(r) for r in responses]

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
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
        
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        return {
            'correctness': reward_scores[0].item(),
            'integer_format': reward_scores[1].item(),
            'strict_format': reward_scores[2].item(),
            'soft_format': reward_scores[3].item(),
            'xml_count': reward_scores[4].item()
        }
