"""
Hold all data sets 

"""

import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from abc import ABC, abstractmethod
from typing import Tuple, Any



class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    This class defines the interface that all dataset loaders should implement.
    Specific dataset loaders should inherit from this class and implement the
    required methods.
    
    Attributes:
        random (bool): If True, returns items randomly; if False, returns sequentially
        current_index (int): Current position for sequential access
    """
    
    def __init__(self, random: bool = False) -> None:
        self.random = random
        self.current_index = 0
        
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        pass
        
    @abstractmethod
    def __iter__(self) -> 'DataLoader':
        """Return self as iterator."""
        return self
        
    @abstractmethod
    def __next__(self) -> Any:
        """Return the next item(s) in the dataset."""
        pass


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()



SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""



class GSM8KLoader(DataLoader):
    """
    A loader class that provides iteration over GSM8K math problems.
    
    This class implements both sequential and random access to math problems through
    standard Python iterator protocols. It can be used to iterate over problems either
    in order or randomly, making it suitable for both training and evaluation.
    
    Attributes:
        questions (List[str]): List of math question strings
        answers (List[str]): List of corresponding answer strings
        random (bool): If True, returns problems randomly; if False, returns sequentially
        current_index (int): Current position in the lists for sequential access
        is_training (bool): If True, cycles infinitely; if False, raises StopIteration when done
    """
    
    def __init__(self, questions: list[str], answers: list[str], random: bool = False, is_training: bool = True) -> None:
        super().__init__(random)
        self.questions = questions
        self.answers = answers
        self.is_training = is_training
        self.pre_prompt = """You will be given a question that involves reasoning. You should reason carefully about the question, then provide your answer.
            It is very important that you put your reasoning process inside <reasoning> tags and your final answer inside <answer> tags, like this:

            
            <reasoning>
            Your step-by-step reasoning process here
            </reasoning>
            <answer>
            Your final answer here
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each answer by immediately starting with <reasoning>. 
            It is is extremely important you answer in this way - do not put any information or text outside of these tags!

            Question: """
        self.system_prompt = SYSTEM_PROMPT
        
    def __len__(self) -> int:
        return len(self.questions)
        
    def __iter__(self) -> 'GSM8KLoader':
        return self
        
    def __next__(self) -> tuple[str, str]:
        if self.current_index >= len(self.questions):
            if self.is_training:
                # For training: cycle back to beginning
                self.current_index = 0
            else:
                # For test: stop iteration
                raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.questions) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        return self.questions[idx], self.answers[idx]

    def reset(self):
        self.current_index = 0 


def build_gsm8k_dataloaders() -> Tuple[GSM8KLoader, GSM8KLoader]: 
    data = load_dataset('openai/gsm8k', 'main')["train"]

    questions = []
    parsed_answers = [] 
    for i in tqdm(range(len(data)), desc="Processing"):
        # Try to get answer - if is None dont use this sample 
        ans = extract_hash_answer(data[i]['answer'])
        if ans is None: 
            continue 
        else:
            questions.append(data[i]['question'])
            parsed_answers.append(ans)

    # Randomly split into train/test sets
    total_samples = len(questions)
    test_size = int(total_samples * 0.01)  # 10% for test set
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_samples), test_size)
    test_indices_set = set(test_indices)
    
    # Convert to numpy arrays for easier indexing
    questions = np.array(questions)
    parsed_answers = np.array(parsed_answers)
    
    # Create boolean mask for test indices
    test_mask = np.zeros(total_samples, dtype=bool)
    test_mask[list(test_indices_set)] = True
    
    # Split using boolean indexing
    test_questions = questions[test_mask]
    test_answers = parsed_answers[test_mask]
    train_questions = questions[~test_mask] 
    train_answers = parsed_answers[~test_mask]

    # Setup data loaders 
    trainloader = GSM8KLoader(train_questions.tolist(), train_answers.tolist(), is_training=True)
    testloader = GSM8KLoader(test_questions.tolist(), test_answers.tolist(), is_training=False)
    
    return trainloader, testloader


def build_math500_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """Build train/test loaders for MATH-500 dataset."""
    data = load_dataset('HuggingFaceH4/MATH-500', split='test')
    
    questions = []
    answers = []
    
    for item in tqdm(data, desc="Processing MATH-500"):
        questions.append(item['problem'])
        answers.append(item['answer'])
    
    # Split 90% train, 10% test
    total_samples = len(questions)
    test_size = int(total_samples * 0.1)
    
    test_indices = random.sample(range(total_samples), test_size)
    test_indices_set = set(test_indices)
    
    questions = np.array(questions)
    answers = np.array(answers)
    
    test_mask = np.zeros(total_samples, dtype=bool)
    test_mask[list(test_indices_set)] = True
    
    test_questions = questions[test_mask]
    test_answers = answers[test_mask]
    train_questions = questions[~test_mask]
    train_answers = answers[~test_mask]

    # # Print length of each 
    # print(f"Train questions: {len(train_questions)}")
    # print(f"Test questions: {len(test_questions)}")
    # print(f"Train answers: {len(train_answers)}")
    # print(f"Test answers: {len(test_answers)}")


    trainloader = GSM8KLoader(train_questions.tolist(), train_answers.tolist(), is_training=True)
    testloader = GSM8KLoader(test_questions.tolist(), test_answers.tolist(), is_training=False)
    
    return trainloader, testloader


def get_dataloaders(dataset_name: str) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load ('gsm8k' or 'math500')
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name.lower() == 'gsm8k':
        return build_gsm8k_dataloaders()
    elif dataset_name.lower() == 'math500':
        return build_math500_dataloaders()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: 'gsm8k', 'math500'")


if __name__ == "__main__": 
    trainloader, testloader = get_dataloaders('gsm8k')