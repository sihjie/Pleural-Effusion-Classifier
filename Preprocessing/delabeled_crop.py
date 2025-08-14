import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def img_crop(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load the image. Check the file path.")
        exit()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    # height, width = 720, 960
    mask = np.zeros((720, 960), dtype=np.uint8)

    # 定義扇形區域的多邊形頂點
    polygon_points = np.array([
        [width//2-105, 90],
        [80, height-200], 
        [230, height-85], 
        [width//2, height-75], 
        [width-230, height-85],
        [width-80, height-200],
        [width//2+105, 90] 
    ])

    if '43' in image_path or '67' in image_path:
        polygon_points = np.array([ # 43, 67
            [370, 118],           
            [180, 540],
            [320, 630], 
            [460, 630], 
            [600, 630],
            [720, 540],
            [525, 118]
        ])
        img = cv2.resize(img, (960, 720))
    
    # 在遮罩上繪製多邊形，填充為白色
    cv2.fillPoly(mask, [polygon_points], 255)

    # print('mask shape:', mask.shape, 'img shape:', img.shape)
    # for deciding polygon_points
    # cv2.imwrite('mask.jpg', mask)
    # mask = cv2.imread('mask.jpg')
    # overlay = cv2.addWeighted(img, 1 - 0.5, mask, 0.5, 0)
    # cv2.imshow('overlay', overlay)
    # cv2.waitKey(0)

    # plt.figure(figsize=(10, 10))
    # plt.subplot(121)
    # plt.imshow(overlay, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(array_mask, cmap='gray')
    # plt.show()

    result = cv2.bitwise_and(img, img, mask=mask)
    print('result shape:', result.shape)

    return result

def delabeled(img):
    lower_white = np.array([210, 210, 210])
    upper_white = np.array([255, 255, 255])

    mask = cv2.inRange(img, lower_white, upper_white)

    result = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return result

if __name__ == "__main__":
    IMAGE_FOLDER = 'C:/Users/ashle/Ashlee/BusLab/Workspace/RawDataset/Ultrasound'
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp'):  
            image_path = os.path.join(IMAGE_FOLDER, filename)

        image_path = r'C:\Users\ashle\Ashlee\BusLab\workkkk\RawDataset\Ultrasound\0032-Echo-1.jpg'
        crop_img = img_crop(image_path)
        delabel_img = delabeled(crop_img)

        cv2.imwrite(f'C:/Users/ashle/Ashlee/BusLab/Workspace/Code/Preprocessing/Cropped/{filename[:4]}_de_crop.jpg',delabel_img)
    # img_crop('C:/Users/ashle/Ashlee/BusLab/Workspace/RawDataset/Ultrasound/0043-Echo-1.jpg')


    

