import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from collections import Counter
from datasets import load_dataset # Added Hugging Face datasets

# --- Configuration ---
SQUARE_CROP_DIM = 256  # Initial square crop dimension
GRID_SIZE = 4          # 4x4 grid
CELL_DIM = SQUARE_CROP_DIM // GRID_SIZE # Should be 64
BANNER_ABS_HEIGHT = 50 # Absolute height of the banner before final resize
FINAL_DIM = 224        # Final output image dimension
DEFAULT_SEED = 42
HF_DATASET_NAME = "Chris1/cityscapes"
HF_DATASET_SPLIT = "train"
BANNER_COLOR = "#1E90FF" # Brighter Blue (DodgerBlue)
CANVAS_BACKGROUND_COLOR = "white"
PADDING_SIZE = 5
PADDING_COLOR = "#F5F5F5" # WhiteSmoke (off-white)

MOTORCYCLE_CLASS_ID = 32
CAR_CLASS_ID = 26
MIN_CLASS_PREVALENCE_THRESHOLD = 0.01 # Target class should be at least 1% of the (resized) image pixels

# Basic Cityscapes class names and IDs (simplified - using trainIds)
# Source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
CITYSCAPES_CLASSES = {
    0: "unlabeled", 1: "ego vehicle", 2: "rectification border", 3: "out of roi", 4: "static",
    5: "dynamic", 6: "ground", 7: "road", 8: "sidewalk", 9: "parking", 10: "rail track",
    11: "building", 12: "wall", 13: "fence", 14: "guard rail", 15: "bridge", 16: "tunnel",
    17: "pole", 18: "polegroup", 19: "traffic light", 20: "traffic sign", 21: "vegetation",
    22: "terrain", 23: "sky", 24: "person", 25: "rider", 26: "car", 27: "truck",
    28: "bus", 29: "caravan", 30: "trailer", 31: "train", 32: "motorcycle", 33: "bicycle",
    255: "ignore" # Typically an ignore label
}
# Filter for classes we might want to use in captchas (more distinct objects)
TARGET_CLASSES = {
    k: v for k, v in CITYSCAPES_CLASSES.items() if v in [
        "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
        "traffic light", "traffic sign", "building", "pole", "vegetation"
    ]
}

class CaptchaGenerator:
    def __init__(self, num_examples_to_select: int = 100, seed: int = DEFAULT_SEED):
        self.num_examples_to_select = num_examples_to_select
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.dataset = None
        self._load_hf_dataset()
        if not self.dataset or len(self.dataset) == 0:
            raise ValueError("Failed to load dataset from Hugging Face or dataset is empty.")
        available_indices = list(range(len(self.dataset)))
        if len(self.dataset) > self.num_examples_to_select:
            self.selected_indices_pool = random.sample(available_indices, self.num_examples_to_select)
        else:
            self.selected_indices_pool = available_indices
        print(f"Initialized CaptchaGenerator with a pool of {len(self.selected_indices_pool)} images from Hugging Face dataset.")
        try:
            self.font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 20) # Adjusted size
            self.font_regular = ImageFont.truetype("DejaVuSans.ttf", 14) # Adjusted size
        except IOError:
            try:
                self.font_bold = ImageFont.truetype("arialbd.ttf", 20)
                self.font_regular = ImageFont.truetype("arial.ttf", 14)
            except IOError:
                print("Warning: DejaVuSans or Arial font not found. Using default PIL font.")
                self.font_bold = ImageFont.load_default()
                self.font_regular = ImageFont.load_default()

    def _load_hf_dataset(self):
        """
        Loads the Cityscapes dataset from Hugging Face.
        """
        try:
            print(f"Loading dataset '{HF_DATASET_NAME}' (split: '{HF_DATASET_SPLIT}') from Hugging Face...")
            self.dataset = load_dataset(HF_DATASET_NAME, split=HF_DATASET_SPLIT)
            print(f"Successfully loaded {len(self.dataset)} examples from {HF_DATASET_SPLIT} split.")
        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            self.dataset = None

    def _get_target_class_info(self, mask_array: np.ndarray, target_class_id: int) -> tuple[int, str, float] | None:
        """Checks for a specific target_class_id presence and returns its info if suitable."""
        if mask_array is None or mask_array.ndim != 2: 
            # print(f"Error or invalid dimensions in _get_target_class_info. Mask shape: {mask_array.shape if mask_array is not None else 'None'}")
            return None
        
        counts = Counter(mask_array.flatten())
        total_pixels = mask_array.size

        class_pixel_count = counts.get(target_class_id, 0)
        if class_pixel_count > 0:
            prevalence_ratio = class_pixel_count / total_pixels
            if prevalence_ratio >= MIN_CLASS_PREVALENCE_THRESHOLD:
                class_name = CITYSCAPES_CLASSES.get(target_class_id, f"class_id_{target_class_id}")
                return target_class_id, class_name, prevalence_ratio
            else:
                # print(f"Class ID {target_class_id} found but prevalence {prevalence_ratio:.4f} is below threshold {MIN_CLASS_PREVALENCE_THRESHOLD}")
                return None 
        return None 

    def _draw_banner_content(self, draw: ImageDraw.ImageDraw, banner_width: int, banner_height: int, 
                             target_class_name: str, font_regular: ImageFont.ImageFont, font_bold: ImageFont.ImageFont):
        line1 = "Select all squares with"
        # Add "a" or "an" appropriately. For simplicity, just "a" for now.
        # A more robust solution would check vowels, but this is a common simplification.
        line2 = f"a {target_class_name}" 

        try: 
            line1_bbox = font_regular.getbbox(line1)
            line1_h = line1_bbox[3] - line1_bbox[1]
            line2_bbox = font_bold.getbbox(line2)
            line2_w = line2_bbox[2] - line2_bbox[0]
            line2_h = line2_bbox[3] - line2_bbox[1]
        except AttributeError: 
            _, line1_h = draw.textsize(line1, font=font_regular)
            line2_w, line2_h = draw.textsize(line2, font=font_bold)
        
        spacing = 5 
        total_text_h = line1_h + line2_h + spacing
        
        y_start_line1 = (banner_height - total_text_h) / 2
        x_line1 = (banner_width - draw.textlength(line1, font=font_regular)) / 2
        draw.text((x_line1, y_start_line1), line1, fill="white", font=font_regular)
        
        y_line2 = y_start_line1 + line1_h + spacing
        x_line2 = (banner_width - line2_w) / 2
        draw.text((x_line2, y_line2), line2, fill="white", font=font_bold)

    def _draw_grid_lines(self, draw: ImageDraw.ImageDraw, grid_origin_x: int, grid_origin_y: int, 
                         grid_area_dim: int, num_cells_per_side: int, 
                         line_color="white", line_width=1):
        """Draws grid lines within a specified square area."""
        cell_dim = grid_area_dim / num_cells_per_side
        for i in range(1, num_cells_per_side):
            # Vertical lines
            x = int(grid_origin_x + i * cell_dim)
            draw.line([(x, grid_origin_y), (x, grid_origin_y + grid_area_dim)], fill=line_color, width=line_width)
            # Horizontal lines
            y = int(grid_origin_y + i * cell_dim)
            draw.line([(grid_origin_x, y), (grid_origin_x + grid_area_dim, y)], fill=line_color, width=line_width)

    def _get_target_squares(self, mask_array_resized: np.ndarray, target_class_id: int) -> list[bool]:
        """
        Determines which grid squares contain the target class on a pre-cropped mask.
        A square is considered to contain the target only if at least 10% of its 
        area is covered by the target class.
        """
        if mask_array_resized is None or mask_array_resized.ndim != 2: # Ensure 2D array
            # print(f"Error or invalid dimensions in _get_target_squares. Mask shape: {mask_array_resized.shape if mask_array_resized is not None else 'None'}")
            return [False] * (GRID_SIZE * GRID_SIZE)
        
        img_height, img_width = mask_array_resized.shape # Should be SQUARE_CROP_DIM x SQUARE_CROP_DIM
        if img_height != SQUARE_CROP_DIM or img_width != SQUARE_CROP_DIM:
            print(f"Warning: Mask array received by _get_target_squares is not {SQUARE_CROP_DIM}x{SQUARE_CROP_DIM}. Actual: {img_height}x{img_width}")

        target_squares = [False] * (GRID_SIZE * GRID_SIZE)
        cell_total_pixels = CELL_DIM * CELL_DIM  # Total pixels in each grid square
        
        for i in range(GRID_SIZE):  # Row
            for j in range(GRID_SIZE):  # Col
                y_start = int(i * CELL_DIM)
                y_end = int((i + 1) * CELL_DIM)
                x_start = int(j * CELL_DIM)
                x_end = int((j + 1) * CELL_DIM)
                square_mask_region = mask_array_resized[y_start:y_end, x_start:x_end]
                
                # Count pixels belonging to target class
                target_pixels = np.sum(square_mask_region == target_class_id)
                
                # Calculate percentage coverage
                coverage_percentage = (target_pixels / cell_total_pixels) * 100
                
                # Square contains target if coverage ≥ 10%
                target_squares[i * GRID_SIZE + j] = coverage_percentage >= 10
                
        return target_squares

    def _draw_solution_markers(self, image_to_draw_on: Image.Image, target_squares: list[bool], 
                               grid_origin_x: int, grid_origin_y: int, 
                               grid_area_dim: int, num_cells_per_side: int):
        draw = ImageDraw.Draw(image_to_draw_on)
        cell_dim = grid_area_dim / num_cells_per_side
        marker_color = "red"
        marker_width = 3 # Adjusted marker width
        padding = int(cell_dim * 0.15) # Adjusted padding

        for idx, is_target in enumerate(target_squares):
            if is_target:
                row = idx // num_cells_per_side
                col = idx % num_cells_per_side
                cell_x1 = grid_origin_x + int(col * cell_dim)
                cell_y1 = grid_origin_y + int(row * cell_dim)
                cell_x2 = grid_origin_x + int((col + 1) * cell_dim)
                cell_y2 = grid_origin_y + int((row + 1) * cell_dim)
                draw.line([(cell_x1 + padding, cell_y1 + padding), (cell_x2 - padding, cell_y2 - padding)], 
                          fill=marker_color, width=marker_width)
                draw.line([(cell_x2 - padding, cell_y1 + padding), (cell_x1 + padding, cell_y2 - padding)], 
                          fill=marker_color, width=marker_width)
        return image_to_draw_on

    def generate_captcha_example_for_class(self, dataset_idx: int, target_class_id: int, target_class_name_override: str) -> tuple[Image.Image | None, Image.Image | None, str | None, list[bool] | None]:
        """Generates a captcha for a specific class ID using a specific dataset index."""
        example = None
        try:
            example = self.dataset[dataset_idx]
            original_pil_image = example['image'].convert("RGB")
            mask_pil_image = example['semantic_segmentation'] 
        except Exception as e:
            keys_available = example.keys() if example else "unknown"
            # print(f"Error accessing data for example {dataset_idx}. Keys: {keys_available}. Error: {e}") # Can be noisy
            return None, None, None, None

        resized_image = original_pil_image.resize((SQUARE_CROP_DIM, SQUARE_CROP_DIM), Image.Resampling.LANCZOS)
        resized_mask_pil = mask_pil_image.resize((SQUARE_CROP_DIM, SQUARE_CROP_DIM), Image.Resampling.NEAREST)
        
        temp_mask_array = np.array(resized_mask_pil)
        if temp_mask_array.ndim == 3:
            resized_mask_array = temp_mask_array[:, :, 0]
        elif temp_mask_array.ndim == 2:
            resized_mask_array = temp_mask_array
        else:
            # print(f"Error: Resized mask array has unexpected dimensions: {temp_mask_array.shape}.")
            return None, None, None, None
        
        # Check if the required target_class_id is present with sufficient prevalence
        # This uses the generalized _get_target_class_info method
        class_info = self._get_target_class_info(resized_mask_array, target_class_id)
        if not class_info:
            # print(f"Target class {target_class_name_override} (ID: {target_class_id}) not prevalent enough in image {dataset_idx}.")
            return None, None, None, None 
        
        # _, actual_class_name, _ = class_info # We'll use target_class_name_override for the prompt
        prompt_text_class = target_class_name_override.lower()

        prompt_canvas_width = SQUARE_CROP_DIM
        prompt_canvas_height = BANNER_ABS_HEIGHT + SQUARE_CROP_DIM
        # Changed canvas background to white
        prompt_canvas = Image.new("RGB", (prompt_canvas_width, prompt_canvas_height), CANVAS_BACKGROUND_COLOR) 
        draw_prompt = ImageDraw.Draw(prompt_canvas)

        banner_rect_coords = (0, 0, prompt_canvas_width, BANNER_ABS_HEIGHT)
        # Using new BANNER_COLOR
        draw_prompt.rectangle(banner_rect_coords, fill=BANNER_COLOR) 
        self._draw_banner_content(draw_prompt, prompt_canvas_width, BANNER_ABS_HEIGHT, 
                                  prompt_text_class, self.font_regular, self.font_bold)

        image_paste_y_offset = BANNER_ABS_HEIGHT
        prompt_canvas.paste(resized_image, (0, image_paste_y_offset))

        self._draw_grid_lines(draw_prompt, 0, image_paste_y_offset, 
                              SQUARE_CROP_DIM, GRID_SIZE)

        target_squares = self._get_target_squares(resized_mask_array, target_class_id)
        solution_canvas = prompt_canvas.copy()
        self._draw_solution_markers(solution_canvas, target_squares, 
                                  0, image_paste_y_offset, 
                                  SQUARE_CROP_DIM, GRID_SIZE)

        # Add padding before final resize
        padded_width = prompt_canvas.width + 2 * PADDING_SIZE
        padded_height = prompt_canvas.height + 2 * PADDING_SIZE

        padded_prompt_image = Image.new("RGB", (padded_width, padded_height), PADDING_COLOR)
        padded_prompt_image.paste(prompt_canvas, (PADDING_SIZE, PADDING_SIZE))

        padded_solution_image = Image.new("RGB", (padded_width, padded_height), PADDING_COLOR)
        padded_solution_image.paste(solution_canvas, (PADDING_SIZE, PADDING_SIZE))

        try:
            final_prompt_image = padded_prompt_image.resize((FINAL_DIM, FINAL_DIM), Image.Resampling.LANCZOS)
            final_solution_image = padded_solution_image.resize((FINAL_DIM, FINAL_DIM), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error during final resize to {FINAL_DIM}x{FINAL_DIM}: {e}")
            return None, None, None, None
            
        return final_prompt_image, final_solution_image, prompt_text_class, target_squares

if __name__ == '__main__':
    try:
        print(f"Attempting to load Cityscapes from Hugging Face: {HF_DATASET_NAME}")
        generator = CaptchaGenerator(num_examples_to_select=500, seed=DEFAULT_SEED) 
        if generator.dataset and len(generator.selected_indices_pool) > 0:
            print("\nGenerating example CAPTCHAs (motorcycle and car)...")
            
            indices_to_try = list(generator.selected_indices_pool)
            random.shuffle(indices_to_try)
            
            generated_examples = 0
            target_classes_to_demo = [
                (MOTORCYCLE_CLASS_ID, CITYSCAPES_CLASSES[MOTORCYCLE_CLASS_ID]),
                (CAR_CLASS_ID, CITYSCAPES_CLASSES[CAR_CLASS_ID])
            ]

            for class_id_to_try, class_name_to_try in target_classes_to_demo:
                if generated_examples >= 2: # Max 2 examples for demo (one of each if possible)
                    break
                found_example_for_this_class = False
                for attempt in range(min(len(indices_to_try), 100)):
                    idx_from_pool = indices_to_try[attempt]
                    prompt_img, solution_img, target_name, _ = \
                        generator.generate_captcha_example_for_class(idx_from_pool, 
                                                                   class_id_to_try, 
                                                                   class_name_to_try)
                    if prompt_img:
                        prompt_filename = f"captcha_prompt_DEMO_{target_name.replace(' ', '_')}.png"
                        solution_filename = f"captcha_solution_DEMO_{target_name.replace(' ', '_')}.png"
                        prompt_img.save(prompt_filename)
                        solution_img.save(solution_filename)
                        print(f"Saved DEMO prompt image to: {prompt_filename}")
                        print(f"Saved DEMO solution image to: {solution_filename}")
                        found_example_for_this_class = True
                        generated_examples += 1
                        # Remove used index to avoid trying it again for the other class in this demo
                        indices_to_try.pop(attempt) 
                        break 
                if not found_example_for_this_class:
                    print(f"Could not generate a DEMO {class_name_to_try} CAPTCHA after several attempts.")
        else:
            print("CaptchaGenerator could not be initialized successfully.")
    except ValueError as ve:
        print(f"Error during CaptchaGenerator initialization: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    print("\nGenerator Script finished.") 