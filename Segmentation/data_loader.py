import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch
import random

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, augmented_images=None, augmented_masks=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

        self.augmented_images = augmented_images if augmented_images is not None else []
        self.augmented_masks = augmented_masks if augmented_masks is not None else []

    def __len__(self):
        return len(self.image_paths) + len(self.augmented_images)

    def __getitem__(self, idx):
        if idx < len(self.image_paths):
            image_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

        else:   # Augmented images            
            image = self.augmented_images[idx - len(self.image_paths)]
            mask = self.augmented_masks[idx - len(self.image_paths)]

        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((256, 256))(mask)  # 確保標籤大小統一
            mask = transforms.ToTensor()(mask)

        return image, mask, os.path.basename(image_path) if idx < len(self.image_paths) else f"augmented_{idx}"
    
def augment_data(image_paths, mask_paths, augment_factor=4):
    augmented_images = []
    augmented_masks = []

    for img, mask in zip(image_paths, mask_paths):
        image = Image.open(img).convert('RGB')
        mask = Image.open(mask).convert('L')

        for _ in range(augment_factor - 1):
            # 隨機旋轉
            angle = random.randint(-30, 30)
            rotated_image = image.rotate(angle)
            rotated_mask = mask.rotate(angle)
            
            # 隨機翻轉
            if random.random() > 0.5:
                rotated_image = rotated_image.transpose(Image.FLIP_LEFT_RIGHT)
                rotated_mask = rotated_mask.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 隨機縮放
            scale = random.uniform(0.8, 1.2)
            width, height = rotated_image.size
            new_size = (int(width * scale), int(height * scale))
            scaled_image = rotated_image.resize(new_size)
            scaled_mask = rotated_mask.resize(new_size)
            
            # 添加擴增影像到列表
            augmented_images.append(scaled_image)
            augmented_masks.append(scaled_mask)
    return augmented_images, augmented_masks

def prepare_data(image_dir, mask_dir, test_size=0.2, random_seed=42):
    image_paths = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith(".jpg")]
    mask_paths = [os.path.join(mask_dir, x) for x in os.listdir(mask_dir) if x.endswith(".jpg")]
    
    X_train, X_test, y_train, y_test= train_test_split(image_paths, mask_paths, test_size=test_size, random_state=random_seed)

    transform_for_stats = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
    ])

    # 建立資料集
    train_dataset = SegmentationDataset(X_train, y_train, transform=transform_for_stats)
    test_dataset = SegmentationDataset(X_test, y_test, transform=transform_for_stats)

    return train_dataset, test_dataset