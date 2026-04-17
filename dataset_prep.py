import cv2
import os
import glob
import numpy as np

def prepare_dataset(input_folder, output_folder, target_width=1000, target_height=800):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Grab all JPG and PNG files from the input folder
    image_paths = glob.glob(os.path.join(input_folder, '*.[jp][pn]*[g]')) # matches jpg, jpeg, png
    
    if not image_paths:
        print("No images found in the raw folder! Check your path.")
        return

    print(f"Found {len(image_paths)} images. Starting standardization process...\n")

    for idx, path in enumerate(image_paths):
        # Load the image
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load {path}")
            continue

        original_height, original_width = img.shape[:2]

        # Calculate scaling factors to ensure the image covers the 1000x800 target
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        
        # We use the LARGER ratio to ensure no black borders are left after scaling
        scale_factor = max(width_ratio, height_ratio)
        
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Step 1: Proportionally scale the image
        scaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Step 2: Calculate the exact center coordinates for the crop
        start_x = (new_width - target_width) // 2
        start_y = (new_height - target_height) // 2
        
        # Step 3: Perform the numpy matrix slice (The Center Crop)
        final_img = scaled_img[start_y:start_y + target_height, start_x:start_x + target_width]

        # Save the standardized image
        filename = os.path.basename(path)
        output_path = os.path.join(output_folder, f"standardized_{filename}")
        cv2.imwrite(output_path, final_img)
        
        print(f"[{idx+1}/{len(image_paths)}] Processed: {filename} -> Saved as 1000x800")

    print("\nDataset standardization complete! All images are now perfect 1000x800 matrices.")

if __name__ == "__main__":
    # Define your folder names here
    RAW_FOLDER = "Raw_Downloads"
    STANDARDIZED_FOLDER = "1_Original_Inputs"
    
    prepare_dataset(RAW_FOLDER, STANDARDIZED_FOLDER)