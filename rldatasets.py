"""
Data loader for generating analog clock images and times.
"""

import random
import os
from typing import Tuple, Any, Dict
from abc import ABC, abstractmethod
from clock_generator import TimeObj, ClockGen
from correlation_generator import generate_correlation_plot # Import the correlation generator
from gui_generator import GUIGenerator # Import the new GUI generator
import json # For loading metadata in Captcha loader
from PIL import Image # For loading images in Captcha loader


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

    @abstractmethod
    def reset(self):
        """Reset the iterator to the beginning."""
        pass


# Define the prompt for the Clock dataset
CLOCK_PROMPT = f"""
You will be shown an image of an analog clock. Your task is to determine the time shown.

You must answer in the following format exactly:
<reasoning>
Reason about the positions of the hour, minute, and second hands to determine the time.
</reasoning>
<answer>
[HH:MM:SS]
</answer>
Replace HH, MM, and SS with the two-digit hour (01-12), minute (00-59), and second (00-59).
Do not include any other text in your answer, or any other text after </answer>.

What time is shown on this clock?
"""


class ClockDataLoader(DataLoader):
    """
    A data loader that generates random analog clock images and their corresponding times.
    
    For training, it generates infinitely. For testing, it generates a fixed number.
    
    Attributes:
        dataset_size (int): The nominal size of the dataset (used for testing length).
        is_train (bool): Flag indicating if this is a training loader.
        prompt (str): The instruction prompt for the language model.
        temp_image_path (str): Path where the temporary clock image is saved.
    """
    
    def __init__(self, dataset_size: int = 50, is_train: bool = True) -> None:
        super().__init__(random=True) # Always generates random times
        self.dataset_size = dataset_size
        self.is_train = is_train
        self.prompt = CLOCK_PROMPT
        self.temp_image_path = "temp_clock.png" # Fixed path for the temporary image
        
    def __len__(self) -> int:
        # Return the specified size, mainly relevant for the test set iteration count
        return self.dataset_size
        
    def __iter__(self) -> 'ClockDataLoader':
        self.current_index = 0
        return self
        
    def __next__(self) -> Tuple[str, str]:
        # Stop iteration for the test set after reaching dataset_size
        if not self.is_train and self.current_index >= self.dataset_size:
            raise StopIteration
        
        self.current_index += 1
        
        # Generate a random time
        time_obj = TimeObj()
        
        # Generate the clock image
        clock_gen = ClockGen(time_obj)
        clock_gen.generate_clock(filename=self.temp_image_path)

        # Get the time string in [HH:MM:SS] format
        time_string = str(time_obj)
            
        return self.temp_image_path, time_string

    def reset(self):
        self.current_index = 0 


# --- Correlation Scatter Dataset (using temp file) --- 

CORRELATION_PROMPT = f"""
You will be shown a scatter plot image displaying a relationship between two variables.
Your task is to estimate the Pearson correlation coefficient (R) depicted in the plot.
The correlation R is a value between 0.00 and 1.00, where 0.00 indicates no correlation and 1.00 indicates perfect positive correlation.

You must answer in the following format exactly:
<reasoning>
Analyze the scatter plot. Consider how closely the points follow a linear pattern. A tighter cluster along a line indicates a higher correlation.
</reasoning>
<answer>
X.XX
</answer>
Replace X.XX with your estimated correlation coefficient, formatted to two decimal places (e.g., 0.75, 0.00, 1.00).
Do not include any other text in your answer, or any other text after </answer>.

What is the estimated correlation R shown in this scatter plot?
"""

class CorrelationScatterDataLoader(DataLoader):
    """
    A data loader that generates random correlation scatter plot images and their corresponding R-values,
    saving the image to a temporary file.
    
    Attributes:
        dataset_size (int): The nominal size of the dataset (used for testing length).
        is_train (bool): Flag indicating if this is a training loader.
        prompt (str): The instruction prompt for the language model.
        temp_image_path (str): Path where the temporary correlation image is saved.
        num_points (int): Number of points for the scatter plot.
        image_size (tuple): Size of the generated image.
    """
    
    def __init__(self, dataset_size: int = 50, is_train: bool = True) -> None:
        super().__init__(random=True) # Always generates random correlations
        self.dataset_size = dataset_size
        self.is_train = is_train
        self.prompt = CORRELATION_PROMPT
        self.temp_image_path = "temp_correlation.png" # Fixed path for the temporary image
        self.num_points = 75 # Standard number of points for correlation plots
        self.image_size = (224, 224) # Standard size
        
    def __len__(self) -> int:
        return self.dataset_size
        
    def __iter__(self) -> 'CorrelationScatterDataLoader':
        self.current_index = 0
        return self
        
    def __next__(self) -> Tuple[str, str]: # Returns path and label string
        if not self.is_train and self.current_index >= self.dataset_size:
            raise StopIteration
            
        self.current_index += 1
        
        # Generate a random R value between 0.00 and 1.00
        r_value = random.uniform(0.0, 1.0)
        
        # Generate the scatter plot image and save it to the temp path
        generate_correlation_plot(
            r_value=r_value,
            num_points=self.num_points,
            filename=self.temp_image_path, # Save to temp path
            img_size=self.image_size,
        )

        # Format the R value string to X.XX
        r_string = f"{r_value:.2f}"
            
        return self.temp_image_path, r_string # Return path and label

    def reset(self):
        self.current_index = 0


# --- GUI Interaction Dataset --- 

# Dynamic prompt, base template here. {target_object_name} will be replaced.
GUI_PROMPT_TEMPLATE = """
You will be shown an image of a graphical user interface (GUI).
The image is 224x224 pixels.
Your task is to identify and provide the coordinates to click the target object: '{target_object_name}'.

You must answer in the following format exactly:
<reasoning>
Briefly describe your reasoning for choosing the click location based on the target object's appearance and position.
</reasoning>
<answer>
x,y
</answer>
Replace x and y with the integer pixel coordinates (e.g., 123,45) where you would click for the '{target_object_name}'. The coordinates must be within the 0-223 range for both x and y.
Do not include any other text in your answer, or any other text after </answer>.

Where would you click to interact with the '{target_object_name}'?
"""

class GUIDataLoader(DataLoader):
    """
    A data loader that generates GUI scenes, selects a target object, and provides a prompt to click it.
    
    Attributes:
        dataset_size (int): Nominal size (for testing).
        is_train (bool): Training or testing mode.
        gui_generator (GUIGenerator): Instance of the GUI scene generator.
        temp_image_path (str): Path for the temporary GUI image.
        prompt_template (str): Base prompt string with placeholder for target object name.
        # `self.prompt` will be dynamically set in __next__ for this loader
    """
    def __init__(self, dataset_size: int = 50, is_train: bool = True, 
                 image_width: int = 224, image_height: int = 224, 
                 hard_mode_prob: float = 0.1):
        super().__init__(random=True)
        self.dataset_size = dataset_size
        self.is_train = is_train
        self.gui_generator = GUIGenerator(width=image_width, height=image_height)
        self.temp_image_path = "temp_gui_scene.png"
        self.prompt_template = GUI_PROMPT_TEMPLATE
        self.prompt = "" 
        self.hard_mode_prob = hard_mode_prob

    def __len__(self) -> int:
        return self.dataset_size

    def __iter__(self) -> 'GUIDataLoader':
        self.current_index = 0
        return self

    def __next__(self) -> Tuple[str, Dict[str, Any]]:
        """
        Returns:
            Tuple[str, Dict[str, Any]]: 
                - image_path (str): Path to the saved GUI image.
                - target_info (Dict[str, Any]): Dictionary containing target details:
                    {'name': str, 'bounding_box': tuple, 'center_x': int, 'center_y': int, 'dynamic_prompt': str, 'is_hard': bool}
        """
        if not self.is_train and self.current_index >= self.dataset_size:
            raise StopIteration

        self.current_index += 1
        
        # Determine if this example should be hard mode based on the stored probability
        use_hard_mode = random.random() < self.hard_mode_prob
        
        target_info = None
        max_retries = 5 
        for _ in range(max_retries):
            gui_image, _, temp_target_info = self.gui_generator.generate_scene_with_target(generate_hard_mode=use_hard_mode)
            if temp_target_info:
                target_info = temp_target_info
                break
        
        if not target_info:
            all_elements = json.loads(_)
            if all_elements and any(el['name'] == 'window' for el in all_elements):
                target_info = next(el for el in all_elements if el['name'] == 'window')
            else:
                raise ValueError("Failed to generate a GUI scene with any identifiable target element after multiple retries.")

        gui_image.save(self.temp_image_path)
        
        target_object_name_for_prompt = target_info['name'].replace("_", " ")
        self.prompt = self.prompt_template.format(target_object_name=target_object_name_for_prompt)
        
        # Add the 'is_hard' flag to the answer dictionary
        answer_for_evaluator = {
            "name": target_info["name"],
            "bounding_box": target_info["bounding_box"],
            "center_x": target_info["center_x"],
            "center_y": target_info["center_y"],
            "dynamic_prompt": self.prompt, 
            "is_hard": use_hard_mode
        }
            
        return self.temp_image_path, answer_for_evaluator

    def reset(self):
        self.current_index = 0


# --- Pre-generated CAPTCHA Dataset ---

CAPTCHA_PROMPT_TEMPLATE = """The image displays a CAPTCHA.
Call the tool `click_screen(x,y)` for all squares that contain {target_article} {target_class_name} as suggested in the image.
The image is 224x224 pixels. The top-left corner is (0,0).
Provide your reasoning within <reasoning></reasoning> tags and your answer with tool calls within <answer></answer> tags.
Example of tool call format:
<reasoning>
The square at the top left contains a part of the target.
The square at the bottom right also contains a part of the target.
</reasoning>
<answer>
click_screen(x1,y1)
click_screen(x2,y2)
</answer>
"""

class PreGeneratedCaptchaLoader(DataLoader):
    """
    Loads pre-generated CAPTCHA images and metadata.
    Assumes dataset is in 'captcha_dataset_output' directory structure
    generated by captcha_ds.py.
    """
    def __init__(self, dataset_root_path: str = "captcha_dataset_output", split: str = "train", shuffle: bool = True, preloaded_examples: list = None):
        super().__init__(random=shuffle) # Use shuffle for random arg
        self.dataset_root_path = dataset_root_path
        self.split = split # 'train', 'test' - though we'll load all for now and split later if needed
        self.metadata_files = []
        self.base_img_path = os.path.join(self.dataset_root_path, "images", "input")

        metadata_dir = os.path.join(self.dataset_root_path, "metadata")
        if not os.path.exists(metadata_dir):
            raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

        for fname in os.listdir(metadata_dir):
            if fname.endswith(".json"):
                self.metadata_files.append(os.path.join(metadata_dir, fname))
        
        if not self.metadata_files:
            raise ValueError(f"No metadata JSON files found in {metadata_dir}")

        if self.random:
            random.shuffle(self.metadata_files)
        
        self.current_index = 0
        # self.prompt is dynamic based on metadata, so no static prompt here

        if preloaded_examples:
            self.metadata_files = preloaded_examples
        else:
            self.metadata_files = self.metadata_files[:]

    def __len__(self) -> int:
        return len(self.metadata_files)

    def __iter__(self) -> 'PreGeneratedCaptchaLoader':
        self.current_index = 0
        if self.random:
            random.shuffle(self.metadata_files)
        return self

    def __next__(self) -> Tuple[str, Dict[str, Any]]:
        if self.current_index >= len(self.metadata_files):
            raise StopIteration

        metadata_path = self.metadata_files[self.current_index]
        self.current_index += 1

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        input_image_rel_path = metadata["input_image_path"]
        # Construct absolute path for the image
        # The input_image_path in metadata is relative to output_base_dir.
        # Here, self.dataset_root_path is output_base_dir.
        input_image_abs_path = os.path.join(self.dataset_root_path, input_image_rel_path)
        solution_image_rel_path = metadata["solution_image_path"]
        solution_image_abs_path = os.path.join(self.dataset_root_path, solution_image_rel_path)
        
        target_class_name = metadata["target_class_name"] # e.g., "motorcycle", "car"
        target_squares_boolean = metadata["target_squares_boolean"] # list of 16 bools
        
        # Determine if "a" or "an" should be used
        article = "an" if target_class_name.lower()[0] in 'aeiou' else "a"
        
        dynamic_prompt = CAPTCHA_PROMPT_TEMPLATE.format(
            target_article=article,
            target_class_name=target_class_name
        )

        # The answer data for the evaluator
        answer_data = {
            "dynamic_prompt": dynamic_prompt,
            "target_class_name": target_class_name,
            "target_squares_boolean": target_squares_boolean,
            "solution_image_path": solution_image_abs_path, # Added for PDF reporting
            # For consistency with GUI, though not strictly needed by CaptchaEvaluator yet
            "name": f"captcha_{target_class_name}", 
            "bounding_box": None, # No single bbox for multiple squares
            "center_x": None,
            "center_y": None,
            "is_hard": False # No hard mode concept for captchas yet
        }
        
        # For GRPO framework, return image path and the answer_data dict
        return input_image_abs_path, answer_data

    def reset(self):
        self.current_index = 0
        if self.random: # Re-shuffle on reset if in random mode
            random.shuffle(self.metadata_files)


# --- Factory Function --- 

def get_dataloaders(dataset_name: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset ('clock' or 'correlation' or 'gui' or 'captcha').
        **kwargs: Additional arguments for specific data loaders (e.g., dataset_size, image_width, image_height).

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
        
    Raises:
        ValueError: If dataset_name is not supported.
    """
    dataset_size = kwargs.get('dataset_size', 50)
    image_width = kwargs.get('image_width', 224)
    image_height = kwargs.get('image_height', 224)
    # Get hard mode probability from kwargs, default to 0.1 for training
    hard_mode_prob_train = kwargs.get('hard_mode_prob_train', 0.1)
    # Default hard mode prob for testing is 0.0 unless specified
    hard_mode_prob_test = kwargs.get('hard_mode_prob_test', 0.1) # Let's use same prob for test too

    dataset_name = dataset_name.lower()

    if dataset_name == 'clock':
        trainloader = ClockDataLoader(dataset_size=dataset_size * 100, is_train=True)
        testloader = ClockDataLoader(dataset_size=dataset_size, is_train=False)
        return trainloader, testloader
    elif dataset_name == 'correlation':
        trainloader = CorrelationScatterDataLoader(dataset_size=dataset_size * 100, is_train=True)
        testloader = CorrelationScatterDataLoader(dataset_size=dataset_size, is_train=False)
        return trainloader, testloader
    elif dataset_name == 'gui':
        trainloader = GUIDataLoader(dataset_size=dataset_size * 100, is_train=True, 
                                  image_width=image_width, image_height=image_height, 
                                  hard_mode_prob=hard_mode_prob_train)
        testloader = GUIDataLoader(dataset_size=dataset_size, is_train=False, 
                                 image_width=image_width, image_height=image_height, 
                                 hard_mode_prob=hard_mode_prob_test)
        return trainloader, testloader
    elif dataset_name == 'captcha':
        captcha_dataset_path = kwargs.get('captcha_dataset_path', "captcha_dataset_output")

        # First, load all metadata file paths
        full_captcha_loader = PreGeneratedCaptchaLoader(dataset_root_path=captcha_dataset_path, shuffle=True)
        all_metadata_paths = full_captcha_loader.metadata_files[:]  # Get all metadata paths
        
        # Shuffle the metadata paths
        random.shuffle(all_metadata_paths)
        
        # Calculate split point (75% train, 25% test)
        split_idx = int(len(all_metadata_paths) * 0.75)
        train_metadata_paths = all_metadata_paths[:split_idx]
        test_metadata_paths = all_metadata_paths[split_idx:]
        
        # Create train and test loaders with their respective metadata paths
        trainloader = PreGeneratedCaptchaLoader(dataset_root_path=captcha_dataset_path, split="train", 
                                              shuffle=True, preloaded_examples=train_metadata_paths)
        testloader = PreGeneratedCaptchaLoader(dataset_root_path=captcha_dataset_path, split="test", 
                                             shuffle=False, preloaded_examples=test_metadata_paths)
        
        print(f"Captcha dataset split: {len(train_metadata_paths)} train, {len(test_metadata_paths)} test examples")
        return trainloader, testloader
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Supported: 'clock', 'correlation', 'gui', 'captcha'")


if __name__ == "__main__": 
    # Test Clock Loader
    print("--- Testing Clock Loader ---")
    try:
        train_loader_c, test_loader_c = get_dataloaders('clock', dataset_size=2)
        print(f"Clock Train loader prompt: {train_loader_c.prompt[:80]}...")
        print(f"Clock Test loader length: {len(test_loader_c)}")

        print("  Train Sample:")
        img_path_c, label_c = next(train_loader_c)
        print(f"    Image Path: {img_path_c}, Label: {label_c}")

        print("  Test Samples:")
        test_loader_c.reset()
        for img_path_c, label_c in test_loader_c:
            print(f"    Image Path: {img_path_c}, Label: {label_c}")
        print("  Clock Test loader iteration finished.")
    except Exception as e:
        print(f"Error testing clock loader: {e}")

    # Test Correlation Loader
    print("\n--- Testing Correlation Loader ---")
    try:
        train_loader_r, test_loader_r = get_dataloaders('correlation', dataset_size=2)
        print(f"Correlation Train loader prompt: {train_loader_r.prompt[:80]}...")
        print(f"Correlation Test loader length: {len(test_loader_r)}")

        print("  Train Sample:")
        img_path_r, label_r = next(train_loader_r)
        print(f"    Image Path: {img_path_r}, Label: {label_r}")

        print("  Test Samples:")
        test_loader_r.reset()
        for img_path_r, label_r in test_loader_r:
            print(f"    Image Path: {img_path_r}, Label: {label_r}")
        print("  Correlation Test loader iteration finished.")
    except Exception as e:
        print(f"Error testing correlation loader: {e}")

    # Test GUI Loader
    print("\n--- Testing GUI Loader ---")
    try:
        gui_train_loader, gui_test_loader = get_dataloaders('gui', dataset_size=2)
        print(f"GUI Test loader length: {len(gui_test_loader)}")

        print("  Train Sample (GUI):")
        img_path_gui, target_details_gui = next(gui_train_loader) # Prompt is now part of target_details_gui
        print(f"    Image Path: {img_path_gui}")
        print(f"    Target Name: {target_details_gui['name']}")
        print(f"    Target BBox: {target_details_gui['bounding_box']}")
        print(f"    Dynamic Prompt Snippet: {target_details_gui['dynamic_prompt'][:150]}...")

        print("  Test Samples (GUI):")
        gui_test_loader.reset()
        for i, (img_path_gui, target_details_gui) in enumerate(gui_test_loader):
            print(f"    Test Sample {i+1}:")
            print(f"      Image Path: {img_path_gui}")
            print(f"      Target Name: {target_details_gui['name']}")
            # print(f"      Target BBox: {target_details_gui['bounding_box']}")
            print(f"      Dynamic Prompt Snippet: {target_details_gui['dynamic_prompt'][:150]}...")
            if i == 0: # Plot the first test image with its target for visual check
                example_img = Image.open(img_path_gui)
                plot_data = [{
                    "name": "TARGET_" + target_details_gui['name'], 
                    "bounding_box": target_details_gui['bounding_box'], 
                    "is_truth": True
                }]
                img_with_target_plot = GUIGenerator.plot_predictions(example_img, plot_data, truth_color="blue")
                target_plot_filename = "rldatasets_gui_test_sample_with_target.png"
                img_with_target_plot.save(target_plot_filename)
                print(f"      Saved first test sample with target plotted to: {target_plot_filename}")

        print("  GUI Test loader iteration finished.")
    except Exception as e:
        print(f"Error testing GUI loader: {e}")
        import traceback
        traceback.print_exc()

    # Test Captcha Loader
    print("\n--- Testing CAPTCHA Loader ---")
    # Assume captcha_dataset_output exists from running captcha_ds.py
    if os.path.exists("captcha_dataset_output/metadata"):
        try:
            # Use a small dataset_size for testing the loader, though actual items depend on generated files
            train_loader_cap, test_loader_cap = get_dataloaders('captcha', captcha_dataset_path="captcha_dataset_output")
            
            print(f"Captcha Train loader length: {len(train_loader_cap)}")
            print(f"Captcha Test loader length: {len(test_loader_cap)}")

            print("  Train Sample (CAPTCHA):")
            img_path_cap, answer_data_cap = next(train_loader_cap)
            print(f"    Image Path: {img_path_cap}")
            print(f"    Solution Image Path: {answer_data_cap.get('solution_image_path')}")
            print(f"    Target Class: {answer_data_cap['target_class_name']}")
            print(f"    Target Squares (first 4): {answer_data_cap['target_squares_boolean'][:4]}...")
            print(f"    Dynamic Prompt Snippet: {answer_data_cap['dynamic_prompt'][:150]}...")

            print("  Test Samples (CAPTCHA):")
            test_loader_cap.reset()
            for i, (img_path_cap, answer_data_cap) in enumerate(test_loader_cap):
                if i >= 2: break # Only show first 2 test samples
                print(f"    Test Sample {i+1}:")
                print(f"      Image Path: {img_path_cap}")
                print(f"      Solution Image Path: {answer_data_cap.get('solution_image_path')}")
                print(f"      Target Class: {answer_data_cap['target_class_name']}")
            print("  CAPTCHA Test loader iteration finished (shown 2 samples).")
        except Exception as e:
            print(f"Error testing CAPTCHA loader: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping CAPTCHA loader test: 'captcha_dataset_output/metadata' not found.")
        print("Please run captcha_ds.py first to generate the dataset.")