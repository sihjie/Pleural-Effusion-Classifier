import os
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from PIL import Image
import logging
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import numpy as np
import pandas as pd

from Segmentation.data_loader import prepare_data, SegmentationDataset
from Segmentation.GLFR_main.src.GLFRNet import GLFRNet

def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    transform = checkpoint.get('normalization')
    return model, transform

def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S %p"
    )

def inference(out_dir, model, data_loader, device, accuracies, ious, dices, filenames_list):
    with torch.no_grad():
        for images, masks, filename in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['main_out']
            # predicted = (torch.sigmoid(outputs) > 0.5).type(torch.int32)
            predicted = torch.argmax(outputs, dim=1)

            batch_size = predicted.shape[0]
            for i in range(batch_size):
                predicted_image = predicted[i].cpu().numpy().astype(np.uint8)  # 確保 predicted_image 為 (H, W)
                if predicted_image.ndim == 3:
                    predicted_image = predicted_image[0]
                mask_image = masks[i].cpu().numpy().astype(np.uint8).squeeze()
                file_name = filename[i]
                Image.fromarray(predicted_image * 255, mode='L').resize((960, 720)).save(f'{out_dir}/{file_name.replace("_clahe.jpg", "_predicted.jpg")}')

                y_true = mask_image.flatten()
                y_pred = predicted_image.flatten()

                filenames_list.append(file_name)
                accuracies.append(accuracy_score(y_true, y_pred))
                ious.append(jaccard_score(y_true, y_pred, average='binary'))
                dices.append(f1_score(y_true, y_pred, average='binary'))

def main():
    img_dir = 'Preprocessing/BM3D_CLAHE'
    mask_dir = 'Preprocessing/Mask'
    ckpt_dir = r'Segmentation\results\GLFR___bm3d_clahe_dice\20250424_183429'

    out_dir = 'Segmentation/predicted_masks'
    os.makedirs(out_dir, exist_ok=True)
    pred_mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(pred_mask_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log_file.log")
    setup_logging(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, test_set = prepare_data(img_dir, mask_dir)

    logging.info(f'Model path to inference: {ckpt_dir}')
    logging.info(f'Image directory: {img_dir}')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)


    accuracies = []  
    ious = []
    dices = []
    filenames_list = []
    logging.info("==== starting training set inference ====")
    for fold, (_, val_idx) in enumerate(kf.split(train_set), 1):
        logging.info(f"Fold {fold}] starting inference.-/***")
        logging.info(f"fold {fold} val idx: {val_idx}")

        val_paths = [train_set.image_paths[i] for i in val_idx]
        val_masks = [train_set.mask_paths[i] for i in val_idx]

        print(f"[Fold {fold}] starting inference")
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_fold_{fold}.pth")
        
        model = GLFRNet(n_class=2, backbone='resnet34', aux=True, pretrained_base=True)
        model, transform = load_checkpoint(model, ckpt_path, device)

        val_ds = SegmentationDataset(val_paths, val_masks, transform=transform)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

        inference(pred_mask_dir, model, val_loader, device, accuracies, ious, dices, filenames_list)

        print(f"[Fold {fold}] complete, mask saved at {pred_mask_dir}")
        logging.info(f"[Fold {fold}] complete, predicted mask saved at {pred_mask_dir}\n")

    print("all training set inference done!")
    logging.info("all training set inference done!\n\n")

    logging.info("starting testing set inference")

    best_ckpt_path = os.path.join(ckpt_dir, "ckpt_fold_3.pth")
    logging.info(f"best model path: {best_ckpt_path}")

    model = GLFRNet(n_class=2, backbone='resnet34', aux=True, pretrained_base=True)
    model, transform = load_checkpoint(model, ckpt_path, device)
    test_ds = SegmentationDataset(test_set.image_paths, test_set.mask_paths, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    inference(pred_mask_dir, model, test_loader, device, accuracies, ious, dices, filenames_list)
    logging.info(f"Testing set predicted mask saved at {pred_mask_dir}\n")

    logging.info("Testing set inference done!")

    each_pred_df = pd.DataFrame({
        "filename": filenames_list,
        "dice": dices,
        "iou": ious,
        "accuracy": accuracies,
        })
    each_pred_df = each_pred_df.sort_values(by="filename").reset_index(drop=True)
    logging.info(f"Each image predicted results: \n{each_pred_df}\n")
    each_pred_df.to_csv(os.path.join(out_dir, "each_pred_results.csv"), index=False)

    mean_dice = np.mean(dices)
    mean_iou = np.mean(ious)
    mean_accuracy = np.mean(accuracies)
    logging.info(f"Mean Dice: {mean_dice}, Mean IoU: {mean_iou}, Mean Accuracy: {mean_accuracy}")
    logging.info(f"All images predicted done!")

if __name__ == "__main__":
    main()
