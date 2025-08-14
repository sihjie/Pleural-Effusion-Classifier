import cv2
import numpy as np
import bm3d
import os
import matplotlib.pyplot as plt

def BM3D(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    sigma_psd = 25  # 設定去噪強度（可根據影像調整）
    denoised_image = bm3d.bm3d(image, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    # denoised_image = bm3d.bm3d(image, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)


    return denoised_image

def main():
    IMAGE_FOLDER = "Preprocessing/Cropped"
    TARGET_FOLDER = "Preprocessing/BM3D"
    for filename in os.listdir(IMAGE_FOLDER):
        file_path = os.path.join(IMAGE_FOLDER, filename)
        denoised_image = BM3D(file_path)
        cv2.imwrite(os.path.join(TARGET_FOLDER, filename.replace("_de_crop.jpg", "_despeckle.jpg")), denoised_image)
        print(f"Finish processing {filename}")

if __name__ == "__main__":
    main()

