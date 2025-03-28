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




class DebateDataLoader(DataLoader):
    """
    A loader class that provides iteration over debate topics.
    
    This class implements both sequential and random access to debate topics through
    standard Python iterator protocols. For each topic, it randomly assigns PRO or CON
    position to create debate scenarios.
    """
    
    def __init__(self, topics: list[str], random: bool = False) -> None:
        super().__init__(random)
        self.topics = topics
        self.pre_prompt = """You will be given a debate topic and your position (PRO or CON). You should reason carefully about the position, then provide your argument.
            It is very important that you put your reasoning process inside <reasoning> tags and your final argument inside <answer> tags, like this:

            <reasoning>
            Your step-by-step reasoning process here, considering key points and potential counterarguments
            </reasoning>
            <answer>
            Your clear, concise 2-3 sentence debate position
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each response by immediately starting with <reasoning>. 
            """
        self.system_prompt = SYSTEM_PROMPT  # Using the same system prompt as GSM8K
            
    def __len__(self) -> int:
        return len(self.topics)
        
    def __iter__(self) -> 'DebateDataLoader':
        return self
        
    def __next__(self) -> tuple[str, str]:
        if self.current_index >= len(self.topics):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.topics) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        topic = self.topics[idx]
        position = random.choice(["PRO", "CON"])
        
        # Format the question to include both topic and position
        formatted_question = f"Debate Topic: {topic}\nPosition: {position}"
        
        # The "answer" in this case is the position, which is needed for evaluation
        return formatted_question

    def reset(self):
        self.current_index = 0


def build_debate_dataloaders() -> Tuple[DebateDataLoader, DebateDataLoader]:
    # Define debate topics - non-controversial but engaging topics
    topics = [
        "Video games should be taught as a school sport",
        "All schools should have mandatory cooking classes",
        "Homework should be replaced with project-based learning",
        "Every city should have a night market",
        "Movie theaters should have special quiet showings",
        "All schools should teach sign language",
        "Restaurants should offer smaller portion options",
        "Public spaces should have musical instruments",
        "All high schools should start after 9am",
        "Zoos should focus only on local wildlife",
        "Libraries should have recording studios",
        "Every workplace should allow pets",
        "Schools should teach financial literacy",
        "All restaurants should show calorie counts",
        "Museums should be open late on weekends",
        "Cities should have designated graffiti walls",
        "Schools should teach basic coding",
        "Grocery stores should have recipe stations",
        "All buildings should have rooftop gardens",
        "Cafes should have board game nights",
        "Libraries should offer virtual reality rooms",
        "Parks should have outdoor movie screens",
        "Schools should teach meditation",
        "Restaurants should compost food waste",
        "Cities should have more water fountains",
        "All schools should have maker spaces",
        "Gyms should offer childcare",
        "Libraries should loan art pieces",
        "Hotels should adopt shelter pets",
        "Schools should teach gardening",
        "Airports should have sleeping pods",
        "Malls should have indoor gardens",
        "Restaurants should grow their own herbs",
        "Cities should have free music venues",
        "Schools should teach public speaking",
        "Offices should have nap rooms",
        "Supermarkets should have tasting stations",
        "Libraries should have podcast studios",
        "Parks should have outdoor chess tables",
        "Schools should teach time management",
        "Restaurants should offer cooking classes",
        "Cities should have stargazing areas",
        "Beaches should have free sunscreen",
        "Schools should teach digital citizenship",
        "Hotels should have community spaces",
        "Parks should have fruit trees",
        "Libraries should offer language exchanges",
        "Theaters should have subtitle options",
        "Schools should teach environmental science",
        "Cities should have interactive art installations"
    ]
    # Split into train/test sets (85/15 split)
    total_topics = len(topics)
    test_size = int(total_topics * 0.15)
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_topics), test_size)
    test_indices_set = set(test_indices)
    
    # Split topics
    train_topics = [t for i, t in enumerate(topics) if i not in test_indices_set]
    test_topics = [topics[i] for i in test_indices]
    
    # Create data loaders
    trainloader = DebateDataLoader(train_topics, random=True)
    testloader = DebateDataLoader(test_topics, random=False)
    
    return trainloader, testloader

def get_dataloaders(dataset_name: str) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load ('gsm8k' or 'debate' currently supported)
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name.lower() == 'debate':
        return build_debate_dataloaders()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Currently 'debate' is available.")


if __name__ == "__main__": 
    trainloader, testloader = get_dataloaders('debate')