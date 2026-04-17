import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
import glob

# --- Module 1: Preprocessing ---
def load_image(filepath):
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not find image at '{filepath}'.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return img_rgb, img_gray

# --- Module 2: Energy & Saliency (Protected against float64 crashes) ---
def generate_saliency_map(color_image):
    img_for_saliency = color_image.astype(np.uint8)
    saliency_algorithm = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency_algorithm.computeSaliency(img_for_saliency)
    
    if not success:
        return np.zeros(color_image.shape[:2], dtype=np.float64)
        
    return (saliency_map * 255.0).astype(np.float64)

def calculate_combined_energy(gray_image, color_image, alpha=1.0, beta=3.0):
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_energy = np.abs(sobel_x) + np.abs(sobel_y)
    
    saliency_energy = generate_saliency_map(color_image)
    combined_energy = (alpha * gradient_energy) + (beta * saliency_energy)
    return combined_energy, saliency_energy

# --- Module 3: Forward Energy DP Core (Optimized with Numba) ---
@jit(nopython=True)
def build_forward_dp_table(gray_image, energy_map):
    rows, cols = gray_image.shape
    M = np.zeros((rows, cols), dtype=np.float64)
    backtrack = np.zeros((rows, cols), dtype=np.int32)
    
    for j in range(cols):
        M[0, j] = energy_map[0, j]
    
    for i in range(1, rows):
        for j in range(cols):
            left_val = gray_image[i, j-1] if j > 0 else 0.0
            right_val = gray_image[i, j+1] if j < cols - 1 else 0.0
            up_val = gray_image[i-1, j]
            
            cU = abs(right_val - left_val)
            cL = cU + abs(up_val - left_val)
            cR = cU + abs(up_val - right_val)
            
            if j == 0:
                cost_up = M[i-1, j] + cU
                cost_right = M[i-1, j+1] + cR
                if cost_up <= cost_right:
                    M[i, j] = energy_map[i, j] + cost_up
                    backtrack[i, j] = j
                else:
                    M[i, j] = energy_map[i, j] + cost_right
                    backtrack[i, j] = j + 1
                    
            elif j == cols - 1:
                cost_up = M[i-1, j] + cU
                cost_left = M[i-1, j-1] + cL
                if cost_up <= cost_left:
                    M[i, j] = energy_map[i, j] + cost_up
                    backtrack[i, j] = j
                else:
                    M[i, j] = energy_map[i, j] + cost_left
                    backtrack[i, j] = j - 1
                    
            else:
                cost_left = M[i-1, j-1] + cL
                cost_up = M[i-1, j] + cU
                cost_right = M[i-1, j+1] + cR
                
                min_cost = cost_left
                min_idx = 0
                if cost_up < min_cost:
                    min_cost = cost_up
                    min_idx = 1
                if cost_right < min_cost:
                    min_cost = cost_right
                    min_idx = 2
                    
                M[i, j] = energy_map[i, j] + min_cost
                backtrack[i, j] = j - 1 + min_idx
                
    return M, backtrack

# --- Module 4: Seam Identification & Removal ---
@jit(nopython=True)
def find_seam(dp_matrix, backtrack_matrix):
    rows, cols = dp_matrix.shape
    seam_path = np.zeros(rows, dtype=np.int32)
    
    min_val = dp_matrix[rows - 1, 0]
    min_col_index = 0
    for j in range(1, cols):
        if dp_matrix[rows - 1, j] < min_val:
            min_val = dp_matrix[rows - 1, j]
            min_col_index = j
            
    seam_path[rows - 1] = min_col_index
    
    for i in range(rows - 1, 0, -1):
        parent_col_index = backtrack_matrix[i, seam_path[i]]
        seam_path[i - 1] = parent_col_index
        
    return seam_path

def remove_seam(color_image, gray_image, seam_path):
    rows, cols = gray_image.shape
    mask_gray = np.ones((rows, cols), dtype=bool)
    mask_color = np.ones((rows, cols, 3), dtype=bool)
    
    for i in range(rows):
        mask_gray[i, seam_path[i]] = False
        mask_color[i, seam_path[i], :] = False
        
    new_gray = gray_image[mask_gray].reshape(rows, cols - 1)
    new_color = color_image[mask_color].reshape(rows, cols - 1, 3)
    
    return new_color, new_gray

# --- Module 10: Academic Visualization ---
def generate_algorithm_dashboard(original_color, final_color, energy_map, dp_matrix, save_path):
    plt.figure(figsize=(16, 8))
    plt.suptitle("Content-Aware Retargeting: Forward Energy DP Pipeline", fontsize=18, fontweight='bold')
    
    plt.subplot(1, 4, 1)
    plt.title(f"1. Original Input\n({original_color.shape[1]} x {original_color.shape[0]})")
    plt.imshow(original_color.astype(np.uint8))
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("2. Combined Energy Map\n(Gradient + Saliency)")
    plt.imshow(energy_map, cmap='hot')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("3. Cumulative Cost Matrix\n(Dynamic Programming Table)")
    plt.imshow(dp_matrix, cmap='viridis')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title(f"4. Retargeted Output\n({final_color.shape[1]} x {final_color.shape[0]})")
    plt.imshow(final_color.astype(np.uint8))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close() # Closes the figure memory to prevent loops from crashing

# --- Execution Core ---
if __name__ == "__main__":
    
    # 1. Define your folders
    INPUT_FOLDER = "1_Original_Inputs" # Put your 12 images here
    OUTPUT_FOLDER = "Dashboard_Results" # The script will create this folder
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 2. Grab all images in the input folder
    image_paths = glob.glob(os.path.join(INPUT_FOLDER, '*.[jp][pn]*[g]'))
    
    if not image_paths:
        print(f"ERROR: No images found in '{INPUT_FOLDER}'. Please add your images and try again.")
    else:
        print(f"Found {len(image_paths)} images. Commencing Batch Dashboard Generation...\n")
        
        # 3. Main processing loop for all 12 images
        for idx, IMAGE_PATH in enumerate(image_paths):
            filename = os.path.basename(IMAGE_PATH)
            filename_no_ext = os.path.splitext(filename)[0]
            
            print("="*40)
            print(f"[{idx+1}/{len(image_paths)}] Processing: {filename}")
            print("="*40)
            
            try:
                color_matrix, gray_matrix = load_image(IMAGE_PATH)
                
                # --- THE DEMO-SCALE TRICK ---
                MAX_WIDTH = 1000
                if color_matrix.shape[1] > MAX_WIDTH:
                    scaling_factor = MAX_WIDTH / color_matrix.shape[1]
                    new_dims = (MAX_WIDTH, int(color_matrix.shape[0] * scaling_factor))
                    
                    color_matrix = cv2.resize(color_matrix.astype(np.uint8), new_dims).astype(np.float64)
                    gray_matrix = cv2.resize(gray_matrix.astype(np.uint8), new_dims).astype(np.float64)
                    
                original_color = color_matrix.copy()
                
                # --- TARGET SETTINGS ---
                SEAMS_TO_REMOVE = 400
                print(f"-> Target: Removing {SEAMS_TO_REMOVE} vertical seams.")
                
                last_energy, last_dp = None, None
                
                # --- THE MAIN DP LOOP ---
                for step in range(SEAMS_TO_REMOVE):
                    energy, _ = calculate_combined_energy(gray_matrix, color_matrix)
                    dp, backtrack = build_forward_dp_table(gray_matrix, energy)
                    
                    if step == 0:
                        last_energy = energy
                        last_dp = dp
                        
                    seam = find_seam(dp, backtrack)
                    color_matrix, gray_matrix = remove_seam(color_matrix, gray_matrix, seam)
                    
                    if (step + 1) % 100 == 0:
                        print(f"-> Progress: {step + 1}/{SEAMS_TO_REMOVE} seams carved.")
                        
                print("-> Core computation complete.")
                
                # --- EXPORT DASHBOARD ---
                dashboard_save_path = os.path.join(OUTPUT_FOLDER, f"dashboard_{filename_no_ext}.png")
                print(f"-> Generating Matplotlib Dashboard...")
                
                generate_algorithm_dashboard(original_color, color_matrix, last_energy, last_dp, dashboard_save_path)
                
                print(f"✅ Saved correctly to: {dashboard_save_path}\n")
                
            except Exception as e:
                print(f"❌ CRITICAL ERROR ON {filename}: {e}\n")
                
        print("🎉 Batch processing complete! Check the 'Dashboard_Results' folder for all your generated images.")