"""
Hold all data sets 

"""

import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from abc import ABC, abstractmethod
from typing import Tuple, Any, List



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


class LDDataLoader(DataLoader):
    """
    A loader class that provides iteration over subjects for Larry David-style roasts.
    
    This class implements both sequential and random access to roast subjects through
    standard Python iterator protocols. For each subject, it generates a prompt for a
    Larry David-style roast.
    """
    
    def __init__(self, subjects: list[str], random: bool = False) -> None:
        super().__init__(random)
        self.subjects = subjects
        self.pre_prompt = """You will be given a subject to mock in Larry David's style from Curb Your Enthusiasm. 
            You should first reason about how to make it funny and creative, then provide your mocking.
            It is very important that you put your reasoning process inside <reasoning> tags and your actual roast inside <answer> tags, like this:

            <reasoning>
            Your step-by-step reasoning process here, thinking about what makes Larry David's humor unique and how to apply it to this subject
            </reasoning>
            <answer>
            [Speak in first person as Larry David himself]
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each response by immediately starting with <reasoning>. 
            """
        self.system_prompt = SYSTEM_PROMPT
            
    def __len__(self) -> int:
        return len(self.subjects)
        
    def __iter__(self) -> 'LDDataLoader':
        return self
        
    def __next__(self) -> str:
        if self.current_index >= len(self.subjects):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.subjects) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        subject = self.subjects[idx]
        
        # Format the question to include the subject
        formatted_question = f"Roast Subject: {subject}"
        
        return formatted_question

    def reset(self):
        self.current_index = 0


def build_ld_dataloaders() -> Tuple[LDDataLoader, LDDataLoader]:
    # Define roast subjects - mix of celebrities, social norms, and everyday situations
    subjects = [
        "People who clap when planes land",
        "Selfie sticks",
        "People who talk in movie theaters",
        "Social media influencers",
        "People who wear pajamas to the airport",
        "Reality TV shows",
        "People who take phone calls on speaker in public",
        "Food delivery apps",
        "People who post their entire lives on social media",
        "Modern art",
        "People who use emojis in work emails",
        "Smartphone addiction",
        "People who take photos of their food",
        "Online dating",
        "People who wear masks while driving alone",
        "Social media challenges",
        "People who use voice messages",
        "Modern architecture",
        "People who clap at the end of movies",
        "Food trucks",
        "People who use hashtags in regular conversation",
        "Modern music",
        "People who take selfies at funerals",
        "Social media filters",
        "People who use speakerphone in restaurants",
        "Modern fashion",
        "People who post workout videos",
        "Food delivery fees",
        "People who use voice assistants in public",
        "Modern technology",
        "People who take photos of everything",
        "Social media algorithms",
        "People who use emojis in texts",
        "Modern entertainment",
        "People who post their entire lives",
        "Food delivery services",
        "People who use voice messages",
        "Modern society",
        "People who take photos of everything",
        "Social media culture",
        "People who use emojis in emails",
        "Modern lifestyle",
        "People who post their entire lives",
        "Food delivery apps",
        "People who use voice assistants",
        "Modern culture",
        "People who take photos of everything",
        "Social media trends",
        "People who use emojis in texts",
        "Modern world"
    ]
    
    # Split into train/test sets (85/15 split)
    total_subjects = len(subjects)
    test_size = int(total_subjects * 0.15)
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_subjects), test_size)
    test_indices_set = set(test_indices)
    
    # Split subjects
    train_subjects = [s for i, s in enumerate(subjects) if i not in test_indices_set]
    test_subjects = [subjects[i] for i in test_indices]
    
    # Create data loaders
    trainloader = LDDataLoader(train_subjects, random=True)
    testloader = LDDataLoader(test_subjects, random=False)
    
    return trainloader, testloader


class ChoppedDataLoader(DataLoader):
    """
    A loader class that provides iteration over mystery baskets for Chopped-style recipe generation.
    
    This class implements both sequential and random access to mystery baskets through
    standard Python iterator protocols. For each basket, it generates a prompt for creating
    a creative recipe using all the mystery ingredients.
    """
    
    def __init__(self, baskets: list[list[str]], random: bool = False) -> None:
        super().__init__(random)
        self.baskets = baskets
        self.pre_prompt = """You will be given 4 mystery ingredients from a Chopped-style cooking show. 
            You should first reason about how to create the best possible dish using these ingredients,
            then provide a detailed recipe. It is very important that you put your reasoning process inside 
            <reasoning> tags and your recipe inside <answer> tags, like this:

            <reasoning>
            Your step-by-step reasoning process here, thinking about:
            1. How to balance flavors and textures
            2. Cooking techniques that would work best
            3. How to make the dish cohesive despite unusual combinations
            4. Additional ingredients needed and why
            5. Timing and execution strategy
            </reasoning>
            <answer>
            A detailed recipe that:
            - Uses all mystery ingredients
            - Can be made in 40 minutes
            - Includes a list of additional ingredients needed
            - Has clear step-by-step instructions
            - Includes timing estimates
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! 
            Start each response by immediately starting with <reasoning>. 
            """
        self.system_prompt = SYSTEM_PROMPT
            
    def __len__(self) -> int:
        return len(self.baskets)
        
    def __iter__(self) -> 'ChoppedDataLoader':
        return self
        
    def __next__(self) -> str:
        if self.current_index >= len(self.baskets):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.baskets) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        basket = self.baskets[idx]
        
        # Format the question to include the mystery basket
        formatted_question = f"Mystery Basket:\n" + "\n".join(f"- {ingredient}" for ingredient in basket)
        
        return formatted_question

    def reset(self):
        self.current_index = 0


def build_chopped_dataloaders() -> Tuple[ChoppedDataLoader, ChoppedDataLoader]:
    # Define base ingredients for training - mix of common and unusual ingredients
    train_ingredients = [
        "Gummy bears",
        "Sardines",
        "Popcorn",
        "Hot sauce",
        "Marshmallows",
        "Kimchi",
        "Coffee grounds",
        "Coconut water",
        "Bacon",
        "Blue cheese",
        "Ginger root",
        "Lemon grass",
        "Mint leaves",
        "Pomegranate seeds",
        "Soy sauce",
        "Tofu",
        "Wasabi",
        "Yogurt",
        "Zucchini",
        "Quinoa"
    ]
    
    # Define separate ingredients for testing
    test_ingredients = [
        "Durian",
        "Seaweed",
        "Crickets",
        "Black garlic",
        "Goat cheese",
        "Mango",
        "Curry powder",
        "Pickled ginger",
        "Coconut milk",
        "Tempeh",
        "Star anise",
        "Lime leaves",
        "Cilantro",
        "Pomegranate molasses",
        "Fish sauce",
        "Miso paste",
        "Sesame oil",
        "Kombu",
        "Shiitake mushrooms",
        "Rice vinegar"
    ]
    
    # Generate training baskets (80 baskets)
    train_baskets = []
    for _ in range(80):
        basket = random.sample(train_ingredients, 4)
        train_baskets.append(basket)
    
    # Generate test baskets (20 baskets)
    test_baskets = []
    for _ in range(10):
        basket = random.sample(test_ingredients, 4)
        test_baskets.append(basket)
    
    # Create data loaders
    trainloader = ChoppedDataLoader(train_baskets, random=True)
    testloader = ChoppedDataLoader(test_baskets, random=False)
    
    return trainloader, testloader



# --- SVG DataLoader Implementation ---

class SVGDataLoader(DataLoader):
    """
    A loader class that provides iteration over SVG scene descriptions.

    This class implements sequential and random access to scene descriptions.
    It stores the prompt structure expected by the SVG generation model.
    """

    def __init__(self, scenes: List[str], random: bool = False) -> None:
        super().__init__(random)
        self.scenes = scenes
        # Adapted pre_prompt for SVG generation task
        self.pre_prompt = """You are an expert SVG generator. You will be given a description of a scene. Generate the highest quality SVG code possible for the given scene.
It is very important that you put your reasoning process inside <reasoning> tags and your final SVG code inside <answer> tags, like this:

<reasoning>
Your step-by-step reasoning process here for the SVG design (e.g., elements, colors, layout).
</reasoning>
<answer>
The complete SVG code, starting with <svg> and ending with </svg>. Ensure it is valid SVG.
</answer>

All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each response immediately with <reasoning>.
"""
        # Assuming SYSTEM_PROMPT is defined earlier in this file
        self.system_prompt = SYSTEM_PROMPT

    def __len__(self) -> int:
        """Returns the total number of scenes."""
        return len(self.scenes)

    def __next__(self) -> str:
        """Returns the next scene description."""
        if self.current_index >= len(self.scenes):
            raise StopIteration

        if self.random:
            # If random, pick any index without advancing current_index persistently
            idx = random.randint(0, len(self.scenes) - 1)
        else:
            idx = self.current_index
            self.current_index += 1

        scene_description = self.scenes[idx]

        # Return only the scene description.
        # The caller uses loader.pre_prompt and loader.system_prompt
        return scene_description

    def reset(self):
        """Resets the iterator index."""
        self.current_index = 0

    def __iter__(self) -> 'SVGDataLoader':
        return self


def build_svg_dataloaders(test_split_ratio: float = 0.15) -> Tuple[SVGDataLoader, SVGDataLoader]:
    """Builds and returns training and testing SVGDataLoaders."""
    # Define SVG scenes
    scenes = [
        "man in a canoe on a calm lake at sunrise",
        "two children building a sandcastle on a sunny beach",
        "a black cat sleeping peacefully in a sunbeam on a windowsill",
        "a detailed bowl of colourful fruit (apples, bananas, grapes) on a wooden table",
        "a futuristic cityscape at night with flying cars",
        "a knight facing a dragon in front of a castle",
        "a cozy library scene with bookshelves, an armchair, and a reading lamp",
        "astronaut planting a flag on the moon with Earth in the background",
        "a bustling farmers market with stalls of vegetables and people",
        "a serene underwater scene with coral reefs and colourful fish",
        "a robot chef cooking pasta in a modern kitchen",
        "three hot air balloons floating over rolling green hills",
        "a group of penguins huddled together on an iceberg",
        "a steaming cup of coffee on a rustic wooden table next to a book",
        "a silhouette of a lone tree on a hill during sunset",
        "a detailed illustration of the solar system with planets orbiting the sun",
        "a whimsical village of mushroom houses in an enchanted forest",
        "a busy train station platform with people and trains",
        "a hummingbird sipping nectar from a vibrant flower",
        "a stylized representation of a human brain with glowing connections",
        "a traditional Japanese zen garden with raked sand patterns and carefully placed stones",
        "a single red rose with dew drops on its petals",
        "a world map made of interconnected puzzle pieces",
        "a friendly cartoon monster waving hello",
        "a classic red telephone box on a rainy London street",
        "a bonfire crackling on a beach at night under the stars",
        "a majestic eagle soaring over snow-capped mountains",
        "a plate of sushi with chopsticks arranged neatly",
        "a vintage record player spinning a vinyl disc",
        "a greenhouse filled with exotic plants and flowers",
        "a cozy fireplace with logs burning brightly and stockings hung",
        "a hot air balloon floating over rolling green hills at sunrise",
        "a futuristic robot dog playing fetch with a red ball in a park",
        "a bustling city street at night with neon signs reflecting on wet pavement",
        "a tall stack of pancakes with butter melting and syrup dripping down",
        "a sailboat with full sails cruising on a calm blue ocean under a clear sky",
        "an astronaut planting a flag on the moon surface with Earth visible in the dark sky",
        "a colorful coral reef teeming with tropical fish, seaweed, and sea anemones",
        "a detailed steampunk airship with gears and propellers docked at a sky-port",
        "a wizard's cluttered study filled with glowing potions, ancient books, and a crystal ball",
        "a tranquil Japanese zen garden with raked sand, mossy rocks, and a small pagoda",
        "a vintage treasure map showing islands, sea monsters, and a dotted line to an X",
        "a classic red telephone booth on a cobblestone street corner on a rainy day",
        "a modern minimalist kitchen with stainless steel appliances and a bowl of fruit on the counter",
        "a medieval castle with high stone walls and towers perched on a cliff overlooking the sea",
        "a close-up view of a honeybee with detailed wings collecting pollen from a large sunflower",
        "a Formula 1 race car speeding around a sharp corner of a track, showing motion blur",
        "a cutaway view of an active volcano showing the magma chamber and lava flowing upwards",
        "a diagram of the solar system showing the sun and planets with their orbital paths",
        "a knight in shining armor holding a lance, riding a galloping horse through a forest"
    ]
    random.shuffle(scenes) # Shuffle before splitting

    # Split into train/test sets
    total_scenes = len(scenes)
    test_size = int(total_scenes * test_split_ratio)
    train_size = total_scenes - test_size

    if test_size == 0 or train_size == 0:
        if total_scenes > 0:
            train_scenes = scenes
            test_scenes = scenes
        else:
            train_scenes = []
            test_scenes = []
    else:
        test_scenes = scenes[:test_size]
        train_scenes = scenes[test_size:]

    # Create data loaders
    trainloader = SVGDataLoader(train_scenes, random=True)
    testloader = SVGDataLoader(test_scenes, random=False)

    print(f"Created SVG DataLoaders: {len(train_scenes)} train scenes, {len(test_scenes)} test scenes.")
    return trainloader, testloader


def get_dataloaders(dataset_name: str) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load ('gsm8k', 'debate', 'ld', or 'chopped' currently supported)
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name.lower() == 'debate':
        return build_debate_dataloaders()
    elif dataset_name.lower() == 'ld':
        return build_ld_dataloaders()
    elif dataset_name.lower() == 'chopped':
        return build_chopped_dataloaders()
    elif dataset_name.lower() == 'svg':
        return build_svg_dataloaders()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Currently 'debate', 'ld', 'chopped', and 'svg' are available.")


if __name__ == "__main__": 
    trainloader, testloader = get_dataloaders('debate')
