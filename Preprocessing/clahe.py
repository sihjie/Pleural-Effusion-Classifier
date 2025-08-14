import cv2
import numpy as np
import os

def clahe(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    clip_limit = 3.0
    tile_grid_size = (8, 8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    clahe_image = clahe.apply(image)

    return clahe_image


def main():
    IMAGE_FOLDER = "Preprocessing/BM3D"
    TARGET_FOLDER = "Preprocessing/BM3D_CLAHE"
    for filename in os.listdir(IMAGE_FOLDER):
        file_path = os.path.join(IMAGE_FOLDER, filename)
        clahe_image = clahe(file_path)
        cv2.imwrite(os.path.join(TARGET_FOLDER, filename.replace("_despeckle.jpg", "_clahe.jpg")), clahe_image)

if __name__ == "__main__":
    main()

