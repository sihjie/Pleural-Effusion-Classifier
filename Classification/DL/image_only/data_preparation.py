import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import random
import matplotlib.pyplot as plt


class PleuralEffusionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augmented_images=None, augmented_labels=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        self.augmented_images = augmented_images if augmented_images is not None else []
        self.augmented_labels = augmented_labels if augmented_labels is not None else []


    def __len__(self):
        return len(self.image_paths) + len(self.augmented_images)

    def __getitem__(self, idx):
        if idx < len(self.image_paths):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]
        else:
            # 擴增影像
            image = self.augmented_images[idx - len(self.image_paths)]
            label = self.augmented_labels[idx - len(self.image_paths)]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, os.path.basename(image_path) if idx < len(self.image_paths) else f"augmented_{idx}"


def augment_data(image_paths, labels, augment_factor=4):
    augmented_images = []
    augmented_labels = []

    for path, label in zip(image_paths, labels):
        if label == 1:  # 假設 1 是 `malignant`
            
            # 開啟影像
            image = Image.open(path).convert('RGB')
            
            for _ in range(augment_factor - 1):
                # 隨機旋轉
                angle = random.randint(-30, 30)
                rotated_image = image.rotate(angle)

                # 隨機翻轉
                if random.random() > 0.5:
                    rotated_image = rotated_image.transpose(Image.FLIP_LEFT_RIGHT)
                
                # 隨機縮放
                scale = random.uniform(0.8, 1.2)
                width, height = rotated_image.size
                new_size = (int(width * scale), int(height * scale))
                scaled_image = rotated_image.resize(new_size)
                
                # 添加擴增影像到列表
                augmented_images.append(scaled_image)
                augmented_labels.append(label)
    return augmented_images, augmented_labels

def show_augmented_images(augmented_images):
    # 設定要顯示的影像數量
    num_images = min(len(augmented_images), 5)  # 顯示最多 5 張影像
    
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(augmented_images[i])
        plt.axis('off')
        plt.title(f"Augmented Image {i+1}")
    
    plt.show()

def prepare_data(image_folder, csv_file, test_size=0.2, random_seed=42):

    # 讀取臨床資料及影像資料
    data = pd.read_csv(csv_file)
    image_paths = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if x.endswith(".jpg")]
    labels = data['Malignant'].values  # 假設 'label' 欄位存放良性或惡性標籤

    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=test_size, random_state=random_seed, stratify=labels)
    print("Original train dataset size:", len(X_train))

    # 計算資料集的均值和標準差
    transform_for_stats = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
  
    # 建立資料集
    train_dataset = PleuralEffusionDataset(X_train, y_train, transform=transform_for_stats)
    test_dataset = PleuralEffusionDataset(X_test, y_test, transform=transform_for_stats)

    return train_dataset, test_dataset