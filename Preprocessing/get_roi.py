import cv2
import os
import numpy as np

def get_roi(original_folder, mask_folder, output_folder):
    for filename in os.listdir(original_folder):
        original_path = os.path.join(original_folder, filename)
        mask_path = os.path.join(mask_folder, 
                                 filename.replace("clahe.jpg", f"expand_20_inter.jpg"))
        
        original_image = cv2.imread(original_path)
        extend_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if original_image is None or extend_mask is None:
            print(f"Error reading {filename}, skipping.")
            continue

        # 確保 mask 為 uint8，並且值是 0 或 255
        extend_mask = np.array(extend_mask).astype(np.uint8)
        extend_mask = extend_mask * 255 if extend_mask.max() == 1 else extend_mask
        extend_mask = cv2.resize(extend_mask, (original_image.shape[1], original_image.shape[0]))

        prediction_binary = (extend_mask > 127).astype(np.uint8)
        roi_image = cv2.bitwise_and(original_image, original_image, mask=prediction_binary)

        output_path = os.path.join(output_folder, filename.replace("clahe.jpg", f"roi_extend_20.jpg"))
        cv2.imwrite(output_path, roi_image)


original_folder = 'Preprocessing/BM3D_CLAHE'
mask_folder = 'Preprocessing\pixel_expand\inter_window_mask\inter_mask_expand_20'

roi_output_folder = f'Preprocessing/expand_20_roi'
os.makedirs(roi_output_folder, exist_ok=True)

get_roi(original_folder, mask_folder, roi_output_folder)

#     print("Finished processing folder:", folder)

# folder_path = f'Preprocessing/pixel_extend/intersection/inter_mask/roi_extend_15'
# print(folder_path)

# roi_output_folder = f'Preprocessing/pixel_extend/intersection/inter_roi/roi_extend_15'
# os.makedirs(roi_output_folder, exist_ok=True)

# get_roi(original_folder, folder_path, roi_output_folder)

# print("Finished processing folder")

