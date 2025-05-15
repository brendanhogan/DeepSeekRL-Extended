import os
import json
import shutil
import random
from tqdm import tqdm # For progress bar

# Assuming captcha_generator.py is in the same directory or accessible in PYTHONPATH
from captcha_generator import CaptchaGenerator, MOTORCYCLE_CLASS_ID, CAR_CLASS_ID, CITYSCAPES_CLASSES
from captcha_generator import (
    SQUARE_CROP_DIM, GRID_SIZE, CELL_DIM, BANNER_ABS_HEIGHT,
    FINAL_DIM, PADDING_SIZE, DEFAULT_SEED
)

def calculate_final_target_square_coords(target_squares_bool_list: list[bool]) -> list[list[float]]:
    """
    Calculates the bounding box coordinates of target squares on the final 224x224 image.
    Args:
        target_squares_bool_list: A list of 16 booleans indicating if a square is a target.
    Returns:
        A list of [x1, y1, x2, y2] coordinate lists for each target square.
    """
    final_coords_list = []

    # Dimensions of the canvas before the final 224x224 resize, but after padding
    padded_content_width = SQUARE_CROP_DIM 
    padded_content_height = BANNER_ABS_HEIGHT + SQUARE_CROP_DIM
    
    pre_resize_width = padded_content_width + 2 * PADDING_SIZE
    pre_resize_height = padded_content_height + 2 * PADDING_SIZE

    # Offset of the 256x256 grid within the padded pre-resize image
    grid_offset_x_on_padded_img = PADDING_SIZE
    grid_offset_y_on_padded_img = PADDING_SIZE + BANNER_ABS_HEIGHT
    
    # Scale factors from pre-resize (padded) image to final 224x224 image
    scale_x = FINAL_DIM / pre_resize_width
    scale_y = FINAL_DIM / pre_resize_height

    for idx, is_target in enumerate(target_squares_bool_list):
        if is_target:
            row = idx // GRID_SIZE  # 0 to 3
            col = idx % GRID_SIZE   # 0 to 3

            # Top-left of cell within the 256x256 grid (on the unpadded prompt_canvas)
            cell_x1_on_grid = col * CELL_DIM
            cell_y1_on_grid = row * CELL_DIM
            
            # Top-left of cell on the padded pre-resize image
            cell_x1_on_padded = grid_offset_x_on_padded_img + cell_x1_on_grid
            cell_y1_on_padded = grid_offset_y_on_padded_img + cell_y1_on_grid
            
            # Bottom-right of cell on the padded pre-resize image
            cell_x2_on_padded = cell_x1_on_padded + CELL_DIM
            cell_y2_on_padded = cell_y1_on_padded + CELL_DIM

            # Scale to final 224x224 image coordinates
            final_cell_x1 = cell_x1_on_padded * scale_x
            final_cell_y1 = cell_y1_on_padded * scale_y
            final_cell_x2 = cell_x2_on_padded * scale_x
            final_cell_y2 = cell_y2_on_padded * scale_y
            
            final_coords_list.append([final_cell_x1, final_cell_y1, final_cell_x2, final_cell_y2])
            
    return final_coords_list

def generate_captcha_dataset(num_items: int = 100, output_base_dir: str = "captcha_dataset_output", seed: int = DEFAULT_SEED):
    """
    Generates a dataset of CAPTCHA images for motorcycles and cars.
    """
    # Load a larger pool of images to increase chances of finding motorcycles
    # num_examples_to_select in CaptchaGenerator is the size of this initial pool.
    # We might need to try many images from this pool to get num_items motorcycle captchas.
    # Let's make this pool significantly larger than num_items, e.g., 5x or 10x, or even more if motorcycles are rare.
    # The actual number of examples in Cityscapes train split is ~2975. Max pool size is len(dataset).
    initial_pool_size = min(max(num_items * 15, 750), 2975) 
    print(f"Initializing CaptchaGenerator with a pool of up to {initial_pool_size} images (seed {seed})...")
    generator = CaptchaGenerator(num_examples_to_select=initial_pool_size, seed=seed)

    if not generator.dataset or not generator.selected_indices_pool:
        print(f"Could not initialize generator with a pool of images.")
        return

    img_input_dir = os.path.join(output_base_dir, "images", "input")
    img_solution_dir = os.path.join(output_base_dir, "images", "solution")
    metadata_dir = os.path.join(output_base_dir, "metadata")

    if os.path.exists(output_base_dir):
        print(f"Output directory '{output_base_dir}' already exists. Removing it.")
        shutil.rmtree(output_base_dir)
        
    os.makedirs(img_input_dir, exist_ok=True)
    os.makedirs(img_solution_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    print(f"Attempting to generate {num_items} CAPTCHA examples for 'motorcycle' or 'car'...")
    generated_count = 0
    
    # Shuffle the pool of indices to try them in a random order
    indices_to_try = list(generator.selected_indices_pool)
    random.shuffle(indices_to_try) # Use python's random, as generator's np.random is for its internal selections
    
    image_idx_pointer = 0

    target_options = [
        (MOTORCYCLE_CLASS_ID, CITYSCAPES_CLASSES[MOTORCYCLE_CLASS_ID]),
        (CAR_CLASS_ID, CITYSCAPES_CLASSES[CAR_CLASS_ID])
    ]

    with tqdm(total=num_items, desc="Generating CAPTCHAs") as pbar:
        while generated_count < num_items and image_idx_pointer < len(indices_to_try):
            dataset_idx_to_try = indices_to_try[image_idx_pointer]
            
            # Randomly pick a class for this attempt on the current image
            current_target_class_id, current_target_class_name = random.choice(target_options)
            
            # Attempt to generate for the current image and chosen class
            prompt_img, solution_img, target_name_from_gen, target_squares_bools = \
                generator.generate_captcha_example_for_class(
                    dataset_idx_to_try,
                    current_target_class_id,
                    current_target_class_name # Pass the actual name for the prompt
                )

            if prompt_img: # Successful generation for this image and class
                item_id_str = f"{generated_count:04d}"
                # Use target_name_from_gen as it will be lowercase and match the prompt
                safe_target_name = target_name_from_gen.replace(' ', '_') 
                input_img_name = f"{item_id_str}_input_{safe_target_name}.png"
                solution_img_name = f"{item_id_str}_solution_{safe_target_name}.png"
                metadata_name = f"{item_id_str}_metadata_{safe_target_name}.json"
                input_img_path = os.path.join(img_input_dir, input_img_name)
                solution_img_path = os.path.join(img_solution_dir, solution_img_name)
                metadata_path = os.path.join(metadata_dir, metadata_name)

                try:
                    prompt_img.save(input_img_path)
                    solution_img.save(solution_img_path)
                    final_target_coords = calculate_final_target_square_coords(target_squares_bools)
                    
                    # Convert boolean values to integers (0/1) for JSON serialization
                    target_squares_json_safe = [1 if x else 0 for x in target_squares_bools]
                    
                    metadata = {
                        "item_id": item_id_str,
                        "input_image_path": os.path.relpath(input_img_path, output_base_dir),
                        "solution_image_path": os.path.relpath(solution_img_path, output_base_dir),
                        "target_class_name": target_name_from_gen, # Store the name used in prompt
                        "target_class_id": current_target_class_id, # Store the ID too
                        "target_squares_boolean": target_squares_json_safe,
                        "target_square_coords_final": final_target_coords
                    }
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    generated_count += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing or saving item for dataset_idx {dataset_idx_to_try} (target: {target_name_from_gen}): {e}")
            
            # Always move to the next image index, whether this attempt was successful or not.
            # This ensures we try different base images rather than getting stuck on one image
            # if it doesn't contain either target class sufficiently.
            image_idx_pointer += 1

    if generated_count < num_items:
        print(f"\nWarning: Could only generate {generated_count} CAPTCHAs out of the requested {num_items} after trying {image_idx_pointer} images from the pool.")
    else:
        print(f"\nSuccessfully generated {generated_count} items in '{output_base_dir}'.")

if __name__ == '__main__':
    generate_captcha_dataset(num_items=100, seed=42)
    print("Dataset generation script finished.") 