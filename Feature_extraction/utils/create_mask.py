import cv2
import numpy as np

# create mask for choice "origin"
def get_full_mask():
    height, width = 720, 960
    polygon_points = np.array([
        [width//2-105, 90],
        [80, height-200], 
        [230, height-85], 
        [width//2, height-75], 
        [width-230, height-85],
        [width-80, height-200],
        [width//2+105, 90] 
    ])

    mask = np.zeros((720, 960), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 255)
    cv2.imwrite('Feature_extraction/full_selection_mask.jpg', mask)

    polygon_points_spe = np.array([ # 43, 67
        [370, 118],           
        [180, 540],
        [320, 630], 
        [460, 630], 
        [600, 630],
        [720, 540],
        [525, 118]
    ])

    mask_spe = np.zeros((720, 960), dtype=np.uint8)
    cv2.fillPoly(mask_spe, [polygon_points_spe], 255)
    cv2.imwrite('Feature_extraction/full_selection_mask_spe.jpg', mask_spe)

def get_box_mask():
    height, width = 720, 960
    polygon_points = np.array([
        [330, 170],
        [185, 370], 
        [775, 370],
        [640, 170] 
    ])

    mask = np.zeros((720, 960), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 255)
    cv2.imwrite('Feature_extraction/boundingbox_mask.jpg', mask)

    polygon_points_spe = np.array([ # 43, 67
        [350, 170],           
        [240, 405],
        [655, 405],
        [545, 170]
    ])

    mask_spe = np.zeros((720, 960), dtype=np.uint8)
    cv2.fillPoly(mask_spe, [polygon_points_spe], 255)
    cv2.imwrite('Feature_extraction/boundingbox_mask_spe.jpg', mask_spe)

def get_xray_mask():
    img_path = r"C:\Users\ashle\Ashlee\BusLab\workkkk\RawDataset\X-Ray\0045-CXR-1.jpg"
    image = cv2.imread(img_path)

    height, width = 2124, 2124
    polygon_points = np.array([
        [10, 10], 
        [width-10, 10], 
        [width-10, height-10], 
        [10, height-10]
    ])

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 255)
    cv2.imwrite('Feature_extraction/xray_mask.jpg', mask)

    mask_path = 'Feature_extraction/xray_mask.jpg'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (mask.shape[1], mask.shape[0]))
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    mask_overlay = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0, dtype=cv2.CV_8U)
    cv2.imshow('Overlayed Image', mask_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # get_full_mask()
    # get_box_mask()
    get_xray_mask()