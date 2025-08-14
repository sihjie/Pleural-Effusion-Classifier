
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import six
from radiomics import featureextractor
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Choose whether to use ROI or the original image for feature extraction.")
parser.add_argument('--o', type=str, required=True, choices=["roi", "origin", "xray",
                                                             "extend_05", "extend_10", "extend_15", "extend_20", 
                                                             "extend_25", "extend_30", "extend_35", "extend_40",
                                                             "extend_45", "extend_50", "extend_55", "extend_60",
                                                             "extend_65", "extend_70"],
                    help="whether to use ROI or the original image for feature extraction.")
args = parser.parse_args()

# IMG_FOLDER = 'Preprocessing/Cropped'
if args.o == "xray":
    IMG_FOLDER = r'C:\Users\ashle\Ashlee\BusLab\workkkk\RawDataset\X-Ray'
else:
    IMG_FOLDER = 'Preprocessing/BM3D_CLAHE'


if args.o == "roi" or args.o == "origin" or args.o == "xray":
    MASK_FOLDER = r'C:\Users\ashle\Ashlee\BusLab\workkkk\Code\Segmentation\predicted_masks\masks'
else:
    MASK_FOLDER = f'Preprocessing/pixel_expand/inter_window_mask/inter_mask_expand_{args.o[-2:]}'

IMG_CSV = f'Feature_extraction/output_csv/CLAHE_feature.csv'  
# ROI_CSV = f'Feature_extraction/output_csv/roi_CLAHE_feature.csv'
ROI_CSV = f'Feature_extraction/output_csv/roi_bmcl_feature.csv'
XRAY_CSV = f'Feature_extraction/output_csv/xray_feature.csv'
# EXTEND_CSV = f'Feature_extraction/output_csv/roi_extend_{args.o[-2:]}_feature.csv'
EXPAND_CSV = f'Feature_extraction/output_csv/bmcl_expand/roi_bmcl_expand_{args.o[-2:]}_feature.csv'

filenames = []
features = []

if args.o == "roi":
    csv_path = ROI_CSV
elif args.o == "origin":
    csv_path = IMG_CSV
elif args.o == "xray":
    csv_path = XRAY_CSV
elif 'extend' in args.o:
    csv_path = EXPAND_CSV

for imgname, maskname in zip(sorted(os.listdir(IMG_FOLDER)), sorted(os.listdir(MASK_FOLDER))):
    feature_name = []
    feature_value = []
    img_path = os.path.join(IMG_FOLDER, imgname)
    mask_path = os.path.join(MASK_FOLDER, maskname)

    image = sitk.ReadImage(img_path)
    if args.o == "roi":
        mask = sitk.ReadImage(mask_path) 
    elif args.o == "origin":
        mask = sitk.ReadImage("Feature_extraction/full_selection_mask.jpg")
        if '43' in imgname or '67' in imgname:
            mask = sitk.ReadImage("Feature_extraction/full_selection_mask_spe.jpg")
    elif args.o == "box":
        mask = sitk.ReadImage("Feature_extraction/boundingbox_mask.jpg")
        if '43' in imgname or '67' in imgname:
            mask = sitk.ReadImage("Feature_extraction/boundingbox_mask_spe.jpg")
    elif 'extend' in args.o:
        mask = sitk.ReadImage(mask_path)
    elif args.o == "xray":
        mask = sitk.ReadImage("Feature_extraction/xray_mask.jpg")

        # resize xray image size
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(mask)  # 讓標籤影像匹配 image 的尺寸
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # 避免標籤值改變
        resampler.SetOutputPixelType(image.GetPixelID())  # 保持原始像素類型

        # 進行重採樣
        image = resampler.Execute(image)
    

    # 確保影像是單通道灰度圖
    if image.GetNumberOfComponentsPerPixel() > 1:
        image = sitk.VectorIndexSelectionCast(image, 0)  # 選取第一個通道
    
    mask = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)

    # 將影像轉換為 float32（常見的 radiomics 支持類型）
    image = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.Cast(mask, sitk.sitkUInt8)  # 掩膜通常為整數型

    extractor = featureextractor.RadiomicsFeatureExtractor('Feature_extraction/param.yaml')
    feature_vector = extractor.execute(image, mask)

    if args.o == "roi" or args.o == "origin" or args.o == "xray":
        filenames.append(imgname)
    else:
        filenames.append(maskname)

    # filenames.append(imgname)

    for idx, (key, val) in enumerate(six.iteritems(feature_vector)):
        if idx < 22:    continue
        feature_name.append(key)
        feature_value.append(val)
    features.append(feature_value)

df = pd.DataFrame(features, columns=feature_name)
df.insert(0, "filename", filenames)
df.to_csv(csv_path, index=False)
print(f"Feature extraction completed. The features are saved to {csv_path}.")