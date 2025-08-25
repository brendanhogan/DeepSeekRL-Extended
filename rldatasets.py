"""
Hold all data sets 

"""

import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from abc import ABC, abstractmethod
from typing import Tuple, Any
import pdb
import pandas as pd

import reasoning_gym

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
Respond in the following format only:
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
    """
    
    def __init__(self, questions: list[str], answers: list[str], random: bool = False) -> None:
        super().__init__(random)
        self.questions = questions
        self.answers = answers
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
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.questions) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        return self.questions[idx], self.answers[idx]

    def reset(self):
        self.current_index = 0 


class ReasoningGymLoader(DataLoader):
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
    """
    
    def __init__(self, questions: list[str], answers: list[str], entries: list[dict], random: bool = False) -> None:
        super().__init__(random)
        self.questions = questions
        self.answers = answers 
        self.entries = entries
       
        self.system_prompt = SYSTEM_PROMPT
        
    def __len__(self) -> int:
        return len(self.questions)
        
    def __iter__(self) -> 'ReasoningGymLoader':
        return self
        
    def __next__(self) -> tuple[str, str, dict]:
        if self.current_index >= len(self.questions):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.questions) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        return self.questions[idx], self.answers[idx], self.entries[idx]

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
    trainloader = GSM8KLoader(train_questions.tolist(), train_answers.tolist())
    testloader = GSM8KLoader(test_questions.tolist(), test_answers.tolist())
    
    return trainloader, testloader

def build_math500_dataloaders() -> Tuple[GSM8KLoader, GSM8KLoader]: 
    data = load_dataset('aditjain1980/math500', 'default')["test"]
    print("Shape of data: ", len(data))
    questions = []
    parsed_answers = [] 
    for i in tqdm(range(len(data)), desc="Processing"):
        # Try to get answer - if is None dont use this sample 
        ans = data[i]['answer']
        if ans is None: 
            continue 
        else:
            questions.append(data[i]['question'])
            parsed_answers.append(ans)

    # Randomly split into train/test sets
    total_samples = len(questions)
    test_size = int(total_samples * 0.1)  # 10% for test set
    
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
    trainloader = GSM8KLoader(train_questions.tolist(), train_answers.tolist())
    testloader = GSM8KLoader(test_questions.tolist(), test_answers.tolist())
    #pdb.set_trace()
    return trainloader, testloader


class MBPPLoader(DataLoader):
    def __init__(self, questions: list[str], test_cases: list[list[str]], function_signatures: list[str], random: bool = False) -> None:
        super().__init__(random)
        self.questions = questions
        self.test_cases = test_cases
        self.function_signatures = function_signatures
      
    def __len__(self) -> int:
        return len(self.questions)
        
    def __iter__(self) -> 'MBPPLoader':
        return self
        
    def __next__(self) -> tuple[str, list[str], str]:
        if self.current_index >= len(self.questions):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.questions) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        return self.questions[idx], self.test_cases[idx], self.function_signatures[idx]

    def reset(self):
        self.current_index = 0 




def build_mbpp_dataloaders() -> Tuple[MBPPLoader, MBPPLoader]: 
    train_data = load_dataset('aditjain1980/mbpp', 'default')["train"]
    test_data = load_dataset('aditjain1980/mbpp', 'default')["test"]
    ### take 10% of the data for test
    test_size = int(len(train_data) * 0.1)
    test_indices = random.sample(range(len(train_data)), test_size)
    test_data = test_data.select(test_indices)
    print("Shape of train data: ", len(train_data))
    questions = []
    questions_test = []
    parsed_test_cases = [] 
    parsed_test_cases_test = []
    function_signatures = []
    function_signatures_test = []
    for i in tqdm(range(len(train_data)), desc="Processing train data"):
        # Try to get answer - if is None dont use this sample 
        test_cases = train_data[i]['test_cases']
        if test_cases is None: 
            continue 
        else:
            questions.append(train_data[i]['text'])
            parsed_test_cases.append(test_cases)
            function_signatures.append(train_data[i]['function_signature'])
    for i in tqdm(range(len(test_data)), desc="Processing test data"):
        # Try to get answer - if is None dont use this sample 
        test_cases = test_data[i]['test_cases']
        if test_cases is None: 
            continue 
        else:
            questions_test.append(test_data[i]['text'])
            parsed_test_cases_test.append(test_cases)
            function_signatures_test.append(test_data[i]['function_signature'])

    # Setup data loaders 
    trainloader = MBPPLoader(questions, parsed_test_cases, function_signatures)
    testloader = MBPPLoader(questions_test, parsed_test_cases_test, function_signatures_test)
    
    return trainloader, testloader

def build_leetcode_dataloaders() -> Tuple[DataLoader, DataLoader]:
    data = load_dataset('aditjain1980/leetcode', 'default')["train"]
    test_size = int(len(data) * 0.03)
    test_indices = random.sample(range(len(data)), test_size)
    test_data = data.select(test_indices)
    train_indices = list(set(range(len(data))) - set(test_indices))
    train_data = data.select(train_indices)
    print("Shape of train data: ", len(train_data))
    questions = []
    questions_test = []
    parsed_test_cases = [] 
    parsed_test_cases_test = []
    function_signatures = []
    function_signatures_test = []
    for i in tqdm(range(len(train_data)), desc="Processing train data"):
        # Try to get answer - if is None dont use this sample 
        test_cases = train_data[i]['test_cases']
        if test_cases is None: 
            continue 
        else:
            questions.append(train_data[i]['content'].split("**Example 1:")[0])
            parsed_test_cases.append(test_cases)
            function_signatures.append(train_data[i]['function_signature'])
    for i in tqdm(range(len(test_data)), desc="Processing test data"):
        # Try to get answer - if is None dont use this sample 
        test_cases = test_data[i]['test_cases']
        if test_cases is None: 
            continue 
        else:
            questions_test.append(test_data[i]['content'].split("**Example 1:")[0])
            parsed_test_cases_test.append(test_cases)
            function_signatures_test.append(test_data[i]['function_signature'])

    # Setup data loaders 
    trainloader = MBPPLoader(questions, parsed_test_cases, function_signatures)
    testloader = MBPPLoader(questions_test, parsed_test_cases_test, function_signatures_test)
    
    return trainloader, testloader
def build_reasoning_gym_dataloaders(dataset_name: str, size: int) -> Tuple[DataLoader, DataLoader]:
    reasoning_task = dataset_name.split(".")[0]
    
    if reasoning_task == 'shortest_path':
        data = reasoning_gym.create_dataset(reasoning_task, size=10000,seed=42,p_blocked=0.1,min_rows=3,min_cols=3)
    elif reasoning_task == 'family_relationships':
        data = reasoning_gym.create_dataset(reasoning_task, size=10000,seed=42,min_family_size=8,max_family_size=12)
    elif reasoning_task == 'number_sequence':
        data = reasoning_gym.create_dataset('number_sequence', size=10000,max_complexity=4,max_terms=6,min_terms=4,)
    elif reasoning_task == 'maze':
        data = reasoning_gym.create_dataset(reasoning_task, size=5000,seed=42)
    elif reasoning_task == 'sokoban':
        data = reasoning_gym.create_dataset(reasoning_task, size=1000,seed=42,min_w=3,max_w=5,min_h=3,max_h=5,min_boxes=2,max_boxes=3)
    else:   
        data = reasoning_gym.create_dataset(reasoning_task, size=10000,seed=42)

    test_size = int(len(data) * 0.01)
    test_indices = random.sample(range(len(data)), test_size)
    train_indices = list(set(range(len(data))) - set(test_indices))
    questions = []
    answers = []
    questions_test = []
    answers_test = []
    entries = []
    entries_test = []
    for i in tqdm(train_indices, desc="Processing train data"):
        questions.append(data[i]['question'])
        answers.append(data[i]['answer'])
        entries.append(data[i])
    for i in tqdm(test_indices, desc="Processing test data"):
        questions_test.append(data[i]['question'])
        answers_test.append(data[i]['answer'])
        entries_test.append(data[i])
    trainloader = ReasoningGymLoader(questions, answers, entries=entries)
    testloader = ReasoningGymLoader(questions_test, answers_test, entries=entries_test)
    return trainloader, testloader

def get_dataloaders(dataset_name: str) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load ('gsm8k' currently supported)
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name.lower() == 'gsm8k':
        return build_gsm8k_dataloaders()
    elif dataset_name.lower() == 'mbpp':
        return build_mbpp_dataloaders()
    elif dataset_name.lower() == 'leetcode':
        return build_leetcode_dataloaders()
    elif dataset_name.lower() == 'math500':
        return build_math500_dataloaders()
    elif "reasoning_gym" in dataset_name.lower():
        return build_reasoning_gym_dataloaders(dataset_name,dataset_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Currently 'gsm8k' and 'math500' are available.")



if __name__ == "__main__":
    trainloader, testloader = get_dataloaders('gsm8k')
