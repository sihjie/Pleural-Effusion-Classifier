import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Choose model for overlay the results")
parser.add_argument('--modelpath', type=str, required=True,choices=['GLFR-main', 'DUCKNet', 'UNetPlusPlus'],
                    help="choose the model path")
parser.add_argument('--onimg', type=str, required=True, choices=["CLAHE", "original", "despeckle"])
args = parser.parse_args()

def overlay_images(original_folder, predicted_folder, mask_folder, output_folder, alpha=0.5):
    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍歷原始影像資料夾
    for filename in os.listdir(original_folder):
        # 構建原圖和遮罩影像的完整路徑
        original_path = os.path.join(original_folder, filename)
        predicted_path = os.path.join(predicted_folder, filename.replace("de_crop.jpg", "predicted.jpg"))
        mask_path = os.path.join(mask_folder, filename.replace("de_crop.jpg", "mask.jpg"))

        # 讀取原圖和遮罩影像
        original_image = cv2.imread(original_path)
        predicted_image = cv2.imread(predicted_path)
        mask_image = cv2.imread(mask_path)

        # 檢查影像是否成功讀取
        if original_image is None or predicted_image is None or mask_image is None:
            print(f"Error reading {filename}, skipping.")
            continue
        
        # 將預測和 Ground Truth 轉為灰階並二值化
        predicted_gray = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2GRAY)
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

        prediction_binary = (predicted_gray > 127).astype(np.uint8)
        ground_truth_binary = (mask_gray > 127).astype(np.uint8)
        
        # 創建疊加影像，使用原圖的拷貝
        overlay = original_image.copy()

        # 紅色表示分割結果 (如果 prediction_binary 為 1，則將對應像素設為紅色)
        overlay[prediction_binary == 1] = [0, 0, 255]  # OpenCV 中的順序是 BGR，這裡是紅色

        # 綠色表示 Ground Truth (如果 ground_truth_binary 為 1，則將對應像素設為綠色)
        overlay[ground_truth_binary == 1] = [0, 255, 0]  # 綠色

        # 黃色表示重疊部分
        overlap_pixels = prediction_binary & ground_truth_binary
        overlay[overlap_pixels == 1] = [0, 255, 255]  
        

        # 調整遮罩影像的透明度
        # overlay = cv2.addWeighted(original_image, 1-alpha, red_layer, alpha, 0)
        # overlay = cv2.addWeighted(overlay, 1-alpha, green_layer, alpha, 0)
        final_overlay = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)

        # 保存重疊結果影像
        output_path = os.path.join(output_folder, filename.replace("de_crop.jpg", "overlay.jpg"))
        cv2.imwrite(output_path, final_overlay)
        print(f"Saved overlay image: {output_path}")

def get_roi(original_folder, predicted_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(original_folder):
        original_path = os.path.join(original_folder, filename)
        predicted_path = os.path.join(predicted_folder, filename[:5] + "predicted.jpg")
        original_image = cv2.imread(original_path)
        predicted_mask = cv2.imread(predicted_path, cv2.IMREAD_GRAYSCALE)

        if original_image is None or predicted_mask is None:
            print(f"Error reading {filename}, skipping.")
            continue

        # 確保 mask 為 uint8，並且值是 0 或 255
        predicted_mask = np.array(predicted_mask).astype(np.uint8)
        predicted_mask = predicted_mask * 255 if predicted_mask.max() == 1 else predicted_mask
        predicted_mask = cv2.resize(predicted_mask, (original_image.shape[1], original_image.shape[0]))

        prediction_binary = (predicted_mask > 127).astype(np.uint8)
        roi_image = cv2.bitwise_and(original_image, original_image, mask=prediction_binary)

        output_path = os.path.join(output_folder, filename[:5] + "roi.jpg")
        cv2.imwrite(output_path, roi_image)
        print(f"Saved roi image: {output_path}")


# 設定資料夾路徑
if args.onimg == "CLAHE":
    original_folder = 'Preprocessing/CLAHE'
elif args.onimg == "original":
    original_folder = 'Preprocessing/Cropped'
elif args.onimg == "despeckle":
    original_folder = 'Preprocessing/CLAHE_BM3D'

predicted_folder = f'Segmentation/{args.modelpath}/results/all_predicted/img'
mask_folder = 'Preprocessing/Mask'

overlap_output_folder = f'Segmentation/{args.modelpath}/results/all_overlay'
roi_output_folder = f'Segmentation/{args.modelpath}/results/all_roi'

# 調用函數重疊影像
# overlay_images(original_folder, predicted_folder, mask_folder, overlap_output_folder, alpha=0.4)
get_roi(original_folder, predicted_folder, roi_output_folder)
