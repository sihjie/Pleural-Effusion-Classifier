import cv2
import numpy as np
import os

window_mask = cv2.imread('Preprocessing/pixel_extend/intersection/full_selection_mask.jpg', cv2.IMREAD_GRAYSCALE)
spe_window_mask = cv2.imread('Preprocessing/pixel_extend/intersection/full_selection_mask_spe.jpg', cv2.IMREAD_GRAYSCALE)

base_extend_mask_folder = 'Preprocessing/pixel_extend/origin_extend_mask'
dilation_folders = [folder for folder in os.listdir(base_extend_mask_folder)
                    if folder.startswith('roi_extend')]

for i, folder in enumerate(dilation_folders):
    dilation_folder_path = os.path.join(base_extend_mask_folder, folder)
    out_folder = f'Preprocessing/pixel_extend/intersection/roi_extend_{folder[-2:]}'
    os.makedirs(out_folder, exist_ok=True)

    for extend_mask_filename in os.listdir(dilation_folder_path):
        # print(extend_mask_filename)
        extend_mask_path = os.path.join(dilation_folder_path, extend_mask_filename)
        extend_mask = cv2.imread(extend_mask_path, cv2.IMREAD_GRAYSCALE)
        if '43' in extend_mask_path or '67' in extend_mask_path:
            intersection_mask = cv2.bitwise_and(extend_mask, spe_window_mask)
        else:
            intersection_mask = cv2.bitwise_and(extend_mask, window_mask)

        cv2.imwrite(os.path.join(out_folder, extend_mask_filename.replace('.jpg', '_interwindow.jpg')), intersection_mask)
        print(f"Saved {extend_mask_filename} intersection mask.")

