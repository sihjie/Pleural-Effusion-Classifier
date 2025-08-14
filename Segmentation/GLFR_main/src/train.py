import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from .GLFRNet import GLFRNet
from Segmentation.data_loader import prepare_data, augment_data, SegmentationDataset
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
import argparse

current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser(description="Choose the checkpoint path for testing")
parser.add_argument('--img', type=str, required=True, choices=["CLAHE", "original", "despeckle"])
args = parser.parse_args()

if args.img == "CLAHE":
    IMAGE_DIR = "Preprocessing/CLAHE"
elif args.img == "original":
    IMAGE_DIR = "Preprocessing/Cropped"
elif args.img == "despeckle":
    IMAGE_DIR = "Preprocessing/CLAHE_BM3D"
MASK_DIR = "Preprocessing/Mask"

RESULTS_PATH = "Segmentation/GLFR-main/results"
os.makedirs(RESULTS_PATH, exist_ok=True)
OUTPUT_CSV = os.path.join(RESULTS_PATH, '5-fold_training_results.csv')
LOG_PATH = os.path.join(RESULTS_PATH, 'log', f'{current_time}_{args.img}.log')

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename=LOG_PATH,
    filemode='a',
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
)
logging.info(f"Segmentation training started at {current_time} using GLFRNet.")

EPOCHS = 200
BATCH_SIZE = 8
EARLY_STOPPING = 10
num_folds = 5

logging.info(f"Image type: {args.img}")
logging.info(f"Number of epochs: {EPOCHS}\tBatch size: {BATCH_SIZE}\tEarly stopping: {EARLY_STOPPING}\tNumber of folds: {num_folds}")

def save_checkpoint(state, savepath):
    torch.save(state, savepath)
    logging.info(f"Best model saved in {savepath}")

# 準備資料
full_train_dataset, _ = prepare_data(IMAGE_DIR, MASK_DIR)

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold = 0

fold_num = []
fold_dice = []
fold_iou = []
fold_accuracy = []

logging.info(f"Start {num_folds}-fold cross-validation.")

for train_indices, val_indices in kf.split(full_train_dataset):
    fold += 1
    print(f"Fold {fold}/{num_folds}")
    logging.info(f"\nFold {fold}/{num_folds}{'-'*60}")

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    train_image_paths = [full_train_dataset.image_paths[x] for x in train_indices]
    train_mask_paths = [full_train_dataset.mask_paths[x] for x in train_indices]
    val_image_paths = [full_train_dataset.image_paths[x] for x in val_indices]
    val_mask_paths = [full_train_dataset.mask_paths[x] for x in val_indices]

    augment_images, augment_masks = augment_data(train_image_paths, train_mask_paths)
    logging.info(f"Original amount of training dataset: {len(train_image_paths)}\tAmount of validation dataset: {len(val_image_paths)}\tAugmented images: {len(augment_images)}")


    transform_for_stats = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset_for_stats = SegmentationDataset(image_paths=train_image_paths, mask_paths=train_mask_paths, transform=transform_for_stats)  # 不使用 augment_data 的資料來 normalization
    loader_for_stats = DataLoader(dataset_for_stats, batch_size=BATCH_SIZE, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    for images, *_ in loader_for_stats:
        batch_samples = images.size(0)
        total_images += batch_samples
        mean += images.mean([0, 2, 3]) * batch_samples
        std += images.std([0, 2, 3]) * batch_samples

    mean /= total_images
    std /= total_images
    print(f"Mean: {mean}, Std: {std}")
    logging.info(f"Normalization with Mean: {mean}, Std: {std}")

    transform_normalize = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean[0], mean[1], mean[2]], std=[std[0], std[1], std[2]])
    ])

    train_dataset = SegmentationDataset(train_image_paths, train_mask_paths, transform=transform_normalize, augmented_images=augment_images, augmented_masks=augment_masks)
    val_dataset = SegmentationDataset(val_image_paths,val_mask_paths, transform=transform_normalize)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 模型初始化
    model = GLFRNet(n_class=2, backbone='resnet34', aux=True, pretrained_base=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    best_iou = 0.0
    best_accuracy = 0.0
    best_dice = 0.0
    best_loss = float('inf')
    counter = 0

    # 訓練迴圈
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        # train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.long().to(device)  # CrossEntropyLoss 需要 Long 類型的標籤

            optimizer.zero_grad()
            outputs = model(images)["main_out"]
            loss = criterion(outputs, masks.squeeze(1))  # 去掉單一維度
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # train_loader_tqdm.set_postfix(loss=loss.item())
        
        train_loss = running_loss/len(train_loader)

        model.eval()

        val_loss = 0.0
        accuracies = []
        ious = []
        dices = []
        filenames_list = []


        with torch.no_grad():
            for images, masks, file_names in val_loader:
                images = images.to(device)
                masks = masks.long().to(device)
                outputs = model(images)["main_out"]
                loss = criterion(outputs, masks.squeeze(1))
                val_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1).type(torch.int32)

                batch_size = predicted.shape[0]
                for i in range(batch_size):
                    predicted_image = predicted[i].cpu().numpy().astype(np.uint8)  # 確保 predicted_image 為 (H, W)
                    mask_image = masks[i].cpu().numpy().astype(np.uint8).squeeze()
                    file_name = file_names[i]
                    # Image.fromarray(predicted_image * 255, mode='L').save(f'{RESULTS_PATH}/{file_name.replace("_de_crop.jpg", "_predicted.jpg")}')
                
                    y_true = mask_image.flatten()
                    y_pred = predicted_image.flatten()
                
                    accuracy = accuracy_score(y_true, y_pred)
                    iou = jaccard_score(y_true, y_pred, average='binary')
                    dice = f1_score(y_true, y_pred, average='binary')

                    accuracies.append(accuracy)
                    ious.append(iou)
                    dices.append(dice)
                    filenames_list.append(file_name)
                torch.cuda.empty_cache()

        mean_accuracy = np.mean(accuracies)
        mean_iou = np.mean(ious)
        mean_dice = np.mean(dices)
        
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Average Train Loss: {train_loss:.4f}, Val_loss: {val_loss:.4f}, mean_dice: {mean_dice:.4f}, mean_iou: {mean_iou:.4f}, mean_accuracy: {mean_accuracy:.4f}")

        logging.info(
            f"Epoch [{epoch+1}/{EPOCHS}]\t"
            f"Training loss {train_loss:.4f}\t"
            f"Val loss {val_loss:.4f}\t"
            f"mean_dice {mean_dice:.4f}\t"
            f"mean_iou {mean_iou:.4f}\t"
            f"mean_accuracy {mean_accuracy:.4f}\t"        
        )

        CHECKPOINT_PATH = os.path.join(RESULTS_PATH, 'checkpoint', f'{current_time}_{fold}.pth')

        # Early Stopping 
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_accuracy = mean_accuracy
            best_dice = mean_dice
            counter = 0
            save_checkpoint(    # save the best
                {
                    "epoch": epoch+1, 
                    "state_dict": model.state_dict(),
                    "normalization": transform_normalize, 
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_iou": best_iou,
                }, CHECKPOINT_PATH
            )
        else:
            counter += 1
            print(f"Early stopping counter: {counter}/{EARLY_STOPPING}")
            logging.info(f"Early stopping counter: {counter}/{EARLY_STOPPING}")
            if counter >= EARLY_STOPPING:
                fold_num.append(fold)
                fold_dice.append(best_dice)
                fold_iou.append(best_iou)
                fold_accuracy.append(best_accuracy)
                each_pred_df = pd.DataFrame({
                    "filename": filenames_list,
                    "dice": dices,
                    "iou": ious,
                    "accuracy": accuracies
                })
                logging.info(f"Fold {fold} Validation results: \n{each_pred_df}")
                logging.info(f"{'='*30} Early stopping triggered at epoch {epoch+1} {'='*30}")
                print("Early stopping triggered.")
                break

time_df = pd.DataFrame({
    "img": [args.img],
    "time": [current_time],
})
time_df.index = ['time']
time_df.to_csv(OUTPUT_CSV, mode='a', header=True, index=False)

final_results_df = pd.DataFrame({
    "fold": fold_num,
    "dice": fold_dice,
    "iou": fold_iou,
    "accuracy": fold_accuracy
})
print(final_results_df)
final_results_df.to_csv(OUTPUT_CSV, mode='a', header=True, index=False)
logging.info(f"Results of each fold: \n{final_results_df}")

logging.info(
    f"""Results of mean and std: 
    mean dice: {np.mean(fold_dice):.4f}\t std dice: {np.std(fold_dice):.4f}
    mean iou: {np.mean(fold_iou):.4f}\t std iou: {np.std(fold_iou):.4f} 
    mean accuracy: {np.mean(fold_accuracy):.4f}\t std accuracy: {np.std(fold_accuracy):.4f}"""
)
mean_std_fold_df = pd.DataFrame({
    "mean_dice": [np.mean(fold_dice)],
    "std_dice": [np.std(fold_dice)],
    "mean_iou": [np.mean(fold_iou)],
    "std_iou": [np.std(fold_iou)],
    "mean_accuracy": [np.mean(fold_accuracy)],
    "std_accuracy": [np.std(fold_accuracy)]
})
mean_std_fold_df.index = ['mean']
mean_std_fold_df.to_csv(OUTPUT_CSV, mode='a', header=True, index=False)

logging.info(f"{'-'*20} Training finished {'-'*20}")
print("訓練完成！")