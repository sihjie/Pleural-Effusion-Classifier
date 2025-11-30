import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
    
def create_mask(img):
    r, g, b = 255, 255, 1
    low = np.array([b-50, g-50, r-50])
    high = np.array([b+50, g+50, r+50])
    # 建立遮罩
    mask = cv2.inRange(img, low, high)
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    grey = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(grey,127,255,cv2.THRESH_BINARY)

    return thresh1

def img_crop(image_path):
    img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    # 定義扇形區域的多邊形頂點
    if '43' in image_path or '67' in image_path:
        polygon_points = np.array([ # 43, 67
            [600, 160],           
            [280, 735],
            [500, 840], 
            [720, 850], 
            [940, 840],
            [1165, 735],
            [845, 160]
        ])
    else:        
        polygon_points = np.array([
            [width//2-105, 90],
            [80, height-200], 
            [230, height-85], 
            [width//2, height-75], 
            [width-230, height-85],
            [width-80, height-200],
            [width//2+105, 90] 
        ])
    
    cv2.fillPoly(mask, [polygon_points], 255)
    result = cv2.bitwise_and(img, img, mask=mask)

    return result
            
# def start(self):
#     image_folder = 'C:/Users/ashle/Ashlee/BusLab/Workspace/Preprocessing/Mask'  # Change this to your image folder path
#     image_filenames = os.listdir(image_folder)
    
#     n = 1
#     for image_filename in image_filenames:
#         image_name = image_filename.split('-')[0]
#         filename = image_name + '.jpg'
#         image_path = os.path.join(image_folder, image_filename)
#         img = cv2.imread(image_path)

#         img = self.create_mask(img)

#         if n==43 or n==67:
#             img = self.spe_image_crop(img)
#         else:
#             img = self.image_cropping(img)

#         cv2.imwrite(os.path.join('Preprocessing', 'temp_mask', f'{n}_mask.jpg'), img)
#         n = n + 1
    
if __name__ == '__main__':
    IMAGE_FOLDER = 'C:/Users/ashle/Ashlee/BusLab/Workspace/RawDataset/LabelFilled'
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp'):  
            image_path = os.path.join(IMAGE_FOLDER, filename)

        crop_img = img_crop(image_path)
        mask = create_mask(crop_img)

        cv2.imwrite(f'C:/Users/ashle/Ashlee/BusLab/Workspace/Code/Preprocessing/Mask/{filename[:4]}_mask.jpg', mask)

