import os
import cv2
import numpy as np

base_mask_folder = 'Preprocessing/pixel_extend/origin_extend'
image_folder = 'Preprocessing/Cropped'
predict_mask_folder = 'Segmentation/UNetPlusPlus/results/all_predicted/img'

# dilation_folders = [folder for folder in os.listdir(base_mask_folder)
#                     if folder.startswith('roi_extend')]

dilation_folders = ['roi_extend_05', 'roi_extend_25', 'roi_extend_45', 'roi_extend_65', 'roi_extend_85']

for image_filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_filename)
    image = cv2.imread(image_path)

    pre_mask_path = os.path.join(predict_mask_folder, image_filename.replace("de_crop.jpg", "predicted.jpg"))
    pre_mask = cv2.imread(pre_mask_path)

    # 複製原始影像以便疊加
    overlay_image = image.copy()

    for i, folder in enumerate(dilation_folders):
        dilation_folder_path = os.path.join(base_mask_folder, folder)
        mask_path = os.path.join(dilation_folder_path, image_filename.replace("de_crop.jpg", f"extended_{folder[-2:]}.jpg"))
        # print(mask_path)
        mask = cv2.imread(mask_path)
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_binary = (mask_gray > 127).astype(np.uint8)

        # 將mask轉換為彩色影像（填充為白色）
        color_mask = np.zeros_like(image)
        color_mask[mask_binary == 1] = (255, 255, 255)  # 使用白色填充mask區域

        alpha = 0.5
        temp = cv2.addWeighted(color_mask, alpha, overlay_image, 1 - alpha, 0, overlay_image)

    results = cv2.addWeighted(temp, 0.3, image, 0.7, 0)
    results = cv2.addWeighted(pre_mask, 0.3, results, 0.7, 0)
    # cv2.imshow(f"overlay_image", results)
    # cv2.waitKey(0)
    cv2.imwrite(f'Preprocessing/pixel_extend/overlay_img/{image_filename[:4]}_overlay.jpg', results)
print("Complete!")
