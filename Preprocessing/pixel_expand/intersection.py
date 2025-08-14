import cv2
import numpy as np
import os

window_mask = cv2.imread('Preprocessing/pixel_expand/full_selection_mask.jpg', cv2.IMREAD_GRAYSCALE)
spe_window_mask = cv2.imread('Preprocessing/pixel_expand/full_selection_mask_spe.jpg', cv2.IMREAD_GRAYSCALE)

base_extend_mask_folder = 'Preprocessing/pixel_expand/origin_expand_mask'
dilation_folders = [folder for folder in os.listdir(base_extend_mask_folder)
                    if folder.startswith('mask_expand')]

for i, folder in enumerate(dilation_folders):
    dilation_folder_path = os.path.join(base_extend_mask_folder, folder)
    out_folder = f'Preprocessing/pixel_expand/inter_window_mask/inter_mask_expand_{folder[-2:]}'
    os.makedirs(out_folder, exist_ok=True)

    for expand_mask_filename in os.listdir(dilation_folder_path):
        # print(extend_mask_filename)
        extend_mask_path = os.path.join(dilation_folder_path, expand_mask_filename)
        extend_mask = cv2.imread(extend_mask_path, cv2.IMREAD_GRAYSCALE)
        if '43' in extend_mask_path or '67' in extend_mask_path:
            intersection_mask = cv2.bitwise_and(extend_mask, spe_window_mask)
        else:
            intersection_mask = cv2.bitwise_and(extend_mask, window_mask)

        cv2.imwrite(os.path.join(out_folder, expand_mask_filename.replace('.jpg', '_inter.jpg')), intersection_mask)
        print(f"Saved {expand_mask_filename} intersection mask.")

