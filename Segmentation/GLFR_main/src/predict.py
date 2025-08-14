import torch
from torch.utils.data import DataLoader
from Segmentation.data_loader import SegmentationDataset
from .GLFRNet import GLFRNet
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import numpy as np
import pandas as pd
import os
from PIL import Image
import argparse
import logging

RESULTS_PATH = "Segmentation/GLFR-main/results/all_predicted"

parser = argparse.ArgumentParser(description="Choose parameter for training")
parser.add_argument('--img', type=str, required=True, choices=["CLAHE", "original"])
parser.add_argument('--checkpth', type=str, required=True)
args = parser.parse_args()

LOG_PATH = os.path.join(RESULTS_PATH, f'{args.checkpth}_all_predicted.log')
OUTPUT_CSV = os.path.join(RESULTS_PATH, f'{args.checkpth}_all_predictions.csv')

if args.img == "CLAHE":
    IMAGE_DIR = "Preprocessing/CLAHE"
elif args.img == "original":
    IMAGE_DIR = "Preprocessing/Cropped"
CHECKPOINT_PATH = f'Segmentation/GLFR-main/results/checkpoint/{args.checkpth}.pth'

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename=LOG_PATH, 
    filemode="w", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
)
logging.info(f"Segmentation results of all images started with {args.checkpth}.pth using GLFR-Net.")

image_paths = [os.path.join(IMAGE_DIR, x) for x in os.listdir(IMAGE_DIR) if x.endswith(".jpg")]
if args.img == "CLAHE":
    mask_paths = [x.replace("CLAHE", "Mask").replace("_clahe.jpg", "_mask.jpg") for x in image_paths]
elif args.img == "original":
    mask_paths = [x.replace("Cropped", "Mask").replace("_de_crop.jpg", "_mask.jpg") for x in image_paths]

checkpoint = torch.load(CHECKPOINT_PATH)
transform_normalize = checkpoint.get("normalization")

dataset = SegmentationDataset(image_paths=image_paths, mask_paths=mask_paths, transform=transform_normalize)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

# 加載訓練好的模型
model = GLFRNet(n_class=2, backbone='resnet34', aux=True, pretrained_base=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

accuracies = []
ious = []
dices = []
filenames_list = []


# 評估模型
with torch.no_grad():
    for images, masks, filename in loader:
        images, masks = images.to(device), masks.long().to(device)
        outputs = model(images)["main_out"]
        predicted = torch.argmax(outputs, dim=1).type(torch.int32)

        batch_size = predicted.shape[0]
        for i in range(batch_size):
            predicted_image = predicted[i].cpu().numpy().astype(np.uint8)
            mask_image = masks[i].cpu().numpy().astype(np.uint8).squeeze()
            file_name = filename[i]

            y_true = mask_image.flatten()
            y_pred = predicted_image.flatten()

            filenames_list.append(file_name)
            accuracies.append(accuracy_score(y_true, y_pred))
            ious.append(jaccard_score(y_true, y_pred, average='binary'))
            dices.append(f1_score(y_true, y_pred, average='binary'))

            # predicted_image = predicted.squeeze().cpu().numpy().astype(np.uint8) * 255

            if args.img == "CLAHE":
                Image.fromarray(predicted_image * 255, mode='L').resize((960, 720)).save(f'{RESULTS_PATH}/img/{file_name.replace("_clahe.jpg", "_predicted.jpg")}')
            elif args.img == "original":
                Image.fromarray(predicted_image * 255, mode='L').resize((960, 720)).save(f'{RESULTS_PATH}/img/{file_name.replace("_de_crop.jpg", "_predicted.jpg")}')

mean_accuracy = np.mean(accuracies)
mean_iou = np.mean(ious)
mean_dice = np.mean(dices)

logging.info(f"Predicted Shape: {predicted.shape}\tMean Dice: {mean_dice}, Mean IoU: {mean_iou}, Mean Accuracy: {mean_accuracy}")

each_pred_df = pd.DataFrame({
    "filename": filenames_list,
    "dice": dices,
    "iou": ious,
    "accuracy": accuracies,
})
each_pred_df.to_csv(OUTPUT_CSV, index=False)
logging.info(f"All image prediction results:\n{each_pred_df}")
logging.info(f"Prediction results saved to {OUTPUT_CSV}")

logging.info("Mean results:")
logging.info(
    f"""Results of mean and std: 
    mean dice: {mean_dice:.4f}\t std dice: {np.std(dices):.4f}
    mean iou: {mean_iou:.4f}\t std iou: {np.std(ious):.4f} 
    mean accuracy: {mean_accuracy:.4f}\t std accuracy: {np.std(accuracies):.4f}"""
)
mean_pred_df = pd.DataFrame({
    "mean_dice": [mean_dice],
    "mean_iou": [mean_iou],
    "mean_accuracy": [mean_accuracy],
})
mean_pred_df.index = ['mean']
mean_pred_df.to_csv(OUTPUT_CSV, mode='a', header=True, index=False)

std_pred_df = pd.DataFrame({
    "std_dice": [np.std(dices)],
    "std_iou": [np.std(ious)],
    "std_accuracy": [np.std(accuracies)],
})
std_pred_df.index = ['std']
std_pred_df.to_csv(OUTPUT_CSV, mode='a', header=True, index=False)