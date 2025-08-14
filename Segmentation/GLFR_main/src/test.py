import logging
import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import segmentation_models_pytorch as smp
from PIL import Image

from Segmentation.data_loader import prepare_data, SegmentationDataset
from .GLFRNet import GLFRNet

parser = argparse.ArgumentParser(description="Choose the checkpoint path for testing")
parser.add_argument('--img', type=str, required=True, choices=["CLAHE", "original", "despeckle"])
parser.add_argument('--checkpth', type=str, required=True)
args = parser.parse_args()

if args.img == "CLAHE":
    IMAGE_DIR = "Preprocessing/CLAHE"
elif args.img == "original":
    IMAGE_DIR = "Preprocessing/Cropped"
elif args.img == "despeckle":
    IMAGE_DIR = "Preprocessing/CLAHE_BM3D"

MASK_DIR = "Preprocessing/Mask"
RESULTS_PATH = "Segmentation/GLFR-main/results"
CHECKPOINT_PATH = os.path.join(RESULTS_PATH, 'checkpoint', f'{args.checkpth}.pth')
LOG_PATH = os.path.join(RESULTS_PATH, 'log', f'{args.checkpth}_{args.img}_testing.log')

BATCH_SIZE = 8
num_folds = 5


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename=LOG_PATH, 
    filemode="w", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    )
logging.info(f"Segmentation testing started with {args.checkpth} using GLFR-Net.")

_, testing_set = prepare_data(IMAGE_DIR, MASK_DIR)

model = GLFRNet(n_class=2, backbone='resnet34', aux=True, pretrained_base=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

fold_num = []
fold_dice = []
fold_iou = []
fold_accuracy = []

for fold in range(num_folds):
    checkpoint = torch.load(CHECKPOINT_PATH.replace('.pth', f'_{fold+1}.pth'))
    logging.info(f"{'='*30} FOLD {fold+1} {'='*30}")
    logging.info(f"Loading checkpoint {CHECKPOINT_PATH.replace('.pth', f'_{fold+1}.pth')}")
    model.load_state_dict(checkpoint['state_dict'])
    transform_normalize = checkpoint.get("normalization")
    logging.info(f"Normalization with {transform_normalize}")

    test_dataset = SegmentationDataset(image_paths=testing_set.image_paths, mask_paths=testing_set.mask_paths, transform=transform_normalize)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    
    accuracies = []  
    ious = []
    dices = []
    filenames_list = []


    # 評估模型
    with torch.no_grad():
        for images, masks, filename in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["main_out"]
            predicted = torch.argmax(outputs, dim=1).type(torch.int32)

            batch_size = predicted.shape[0]
            for i in range(batch_size):
                predicted_image = predicted[i].cpu().numpy().astype(np.uint8)  # 確保 predicted_image 為 (H, W)
                if predicted_image.ndim == 3:
                    predicted_image = predicted_image[0]
                mask_image = masks[i].cpu().numpy().astype(np.uint8).squeeze()
                file_name = filename[i]

                y_true = mask_image.flatten()
                y_pred = predicted_image.flatten()

                filenames_list.append(file_name)
                accuracies.append(accuracy_score(y_true, y_pred))
                ious.append(jaccard_score(y_true, y_pred, average='binary'))
                dices.append(f1_score(y_true, y_pred, average='binary'))

    each_pred_df = pd.DataFrame({
        "filename": filenames_list,
        "dice": dices,
        "iou": ious,
        "accuracy": accuracies,
    })
    logging.info(f"Each validation results: \n{each_pred_df}")

    mean_dice = np.mean(dices)
    mean_iou = np.mean(ious)
    mean_accuracy = np.mean(accuracies)
    logging.info(f"Fold {fold+1} - Mean Dice: {mean_dice}, Mean IoU: {mean_iou}, Mean Accuracy: {mean_accuracy}")

    fold_num.append(fold+1)
    fold_dice.append(mean_dice)
    fold_iou.append(mean_iou)
    fold_accuracy.append(mean_accuracy)

fold_results_df = pd.DataFrame({
    "fold": fold_num,
    "dice": fold_dice,
    "iou": fold_iou,
    "accuracy": fold_accuracy,
})
logging.info(f"Fold results: \n{fold_results_df}")

best_dice_idx = fold_results_df["dice"].idxmax()
best_fold = fold_results_df.loc[best_dice_idx, "fold"]
logging.info(
    f"""Results of mean and std: 
    mean dice: {np.mean(fold_dice):.4f}\t std dice: {np.std(fold_dice):.4f}
    mean iou: {np.mean(fold_iou):.4f}\t std iou: {np.std(fold_iou):.4f} 
    mean accuracy: {np.mean(fold_accuracy):.4f}\t std accuracy: {np.std(fold_accuracy):.4f}"""
)
logging.info(f"""Best fold: {best_fold} with Dice: {fold_results_df.loc[best_dice_idx, "dice"]}, IoU: {fold_results_df.loc[best_dice_idx, "iou"]}, Accuracy: {fold_results_df.loc[best_dice_idx, "accuracy"]}""")
logging.info(f"The best checkpoint is {CHECKPOINT_PATH.replace('.pth', f'_{best_fold}.pth')}")
logging.info("Segmentation testing finished.")
    