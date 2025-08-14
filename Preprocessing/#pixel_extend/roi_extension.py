import cv2
import os
import numpy as np

# image_folder = "Preprocessing/BM3D_CLAHE"
mask_folder = r"Segmentation\predicted_masks\masks"

for mask_filename in os.listdir(mask_folder):
    # image_path = os.path.join(image_folder, image_filename)
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    mask_path = os.path.join(mask_folder, mask_filename)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    _, thresholded = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_result = np.zeros_like(mask)
    cv2.drawContours(contours_result, contours, -1, (255), thickness=cv2.FILLED)

    kernel = np.ones((3, 3), np.uint8)  # Change the kernel size to adjust the extension length
    for i in range(5, 100, 5):
        out_folder = f'Preprocessing/pixel_expand/origin_expand_mask/mask_expand_{str(i).zfill(2)}'
        os.makedirs(out_folder, exist_ok=True)
        dilated_result = cv2.dilate(contours_result, kernel, iterations=i)
        
        cv2.imwrite(os.path.join(out_folder, mask_filename.replace("predicted.jpg", f"expand_{str(i).zfill(2)}.jpg")), dilated_result)
        print(f"{i} pixel extended mask saved.")