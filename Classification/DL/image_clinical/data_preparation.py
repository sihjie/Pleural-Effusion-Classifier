import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
import logging

class PleuralEffusionDataset(Dataset):
    def __init__(self, image_paths, labels, clinical_features, transform=None, augmented_images=None, augmented_labels=None,augmented_clinical_features=None):
        self.image_paths = image_paths
        self.labels = labels
        self.clinical_features = clinical_features  
        self.transform = transform

        self.augmented_images = augmented_images if augmented_images is not None else []
        self.augmented_labels = augmented_labels if augmented_labels is not None else []
        self.augmented_clinical_features = augmented_clinical_features if augmented_clinical_features is not None else []

    def __len__(self):
        return len(self.image_paths) + len(self.augmented_images)

    def __getitem__(self, idx):
        if idx < len(self.image_paths):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]
            clinical_feature = self.clinical_features[idx]

        else:
            image = self.augmented_images[idx - len(self.image_paths)]
            label = self.augmented_labels[idx - len(self.image_paths)]
            clinical_feature = self.clinical_features[idx - len(self.image_paths)]

        # clinical_feature = self.clinical_features[idx]
        clinical_feature = torch.tensor(clinical_feature, dtype=torch.float32)  # 將臨床數據轉為 tensor   

        if self.transform:
            image = self.transform(image)
        
        return image, clinical_feature, label, os.path.basename(image_path) if idx < len(self.image_paths) else f"augmented_{idx}"  # 返回檔名

class ClinicalDataPreprocessor:
    def __init__(self, csv, feature_type, drop_columns, timestamp):
        self.data = pd.read_csv(csv)
        self.feature_type = feature_type
        self.drop_columns = drop_columns
        self.ohe_columns = ["Appearance", "Color"]
        self.encoder = OneHotEncoder(sparse_output=False)
        self.logger = logging.getLogger(f'PleuralEffusionTrainer_{timestamp}')

    def preprocess_data(self):
        logging.info("----> Data Preprocessing started.\n")
        logging.info(f"Original data columns:\n{self.data.columns}\n")
        if 'clinical' in self.feature_type:
            self._drop_columns()
            logging.info(f"After dropping columns: \n{self.data.columns}\n")
            self.before_ohe_columns = self.data.columns

            self._data_mapping()

            self.after_ohe_data = self._one_hot_encoding()
            logging.info(f"After One-Hot Encoding: \n{self.after_ohe_data.columns}\n")

            self._fill_missing_values()
        else:   # only CXR
            self.data = self.data.drop(columns=['id'])
            logging.info("""
                         drop 'id' column
                         No data preprocessing required for this dataset because not contained clinical features.
                         """)
            self.before_ohe_columns = self.data.columns
        logging.info(f"---- Data Preprocessing completed. ----\n\n")
        return self.data

    def _drop_columns(self):
        self.data = self.data.drop(columns=self.drop_columns, errors='ignore')
        logging.info(f"Columns dropped: {self.drop_columns}")

    def _data_mapping(self):
        mapping = {
            "Macrophage": {"Few": 0, "Some": 1, "Many": 2},
            "Mesothel cell": {"Few": 0, "Some": 1, "Many": 2},
            "Fluid Status": {"Exudate": 1, "Transudate": 0},
            "Protein": {"Negative": 0, "Positive": 1},
        }
        logging.info(f"Data Mapping: \n{mapping}")
        for column, map_values in mapping.items():
            if column in self.data.columns:
                self.data[column] = self.data[column].map(map_values)

    def _one_hot_encoding(self):
        encoded_ohe = self.encoder.fit_transform(self.data[self.ohe_columns])
        encoded_ohe_df = pd.DataFrame(encoded_ohe, columns=self.encoder.get_feature_names_out(self.ohe_columns))
        self.data = pd.concat([self.data.drop(columns=self.ohe_columns), encoded_ohe_df], axis=1)
        logging.info(f"One-Hot Encoding columns: {self.ohe_columns}\tencoded as below:")
        logging.info(f"{self.encoder.get_feature_names_out(self.ohe_columns)}")
        return self.data

    def _fill_missing_values(self):
        columns_to_fill = ['age', 'Macrophage', 'Mesothel cell', 'Glucose (PL)', 
                            'T-Protein (PL)', 'LDH (PL)', 'ADA']
        self.data.loc[:, columns_to_fill] = self.data.loc[:, columns_to_fill].fillna(self.data[columns_to_fill].median())
        logging.info(f"Missing values filled with median: {columns_to_fill}")

def augment_data(image_paths, labels, clinical_features, augment_factor=4):
    augmented_images = []
    augmented_labels = []
    augmented_clinical_features = []

    for path, label, clinical_feature in zip(image_paths, labels, clinical_features):
        if label == 1:  # 假設 1 是 `malignant`
            
            # 擴增影像
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
                
                # 儲存擴增影像
                # augmented_image_path = f"{os.path.splitext(path)[0]}_aug_{_}.jpg" # for croppedImage
                # scaled_image.save(augmented_image_path)
                
                # 添加擴增影像到列表
                augmented_images.append(scaled_image)
                augmented_labels.append(label)
                augmented_clinical_features.append(clinical_feature)

    return augmented_images, augmented_labels, augmented_clinical_features


def prepare_data(image_folder, csv, feature_type, drop_columns, timestamp, test_size=0.2, random_seed=42):

    data_preprocessor = ClinicalDataPreprocessor(csv, feature_type, drop_columns, timestamp)
    data = data_preprocessor.preprocess_data()

    image_paths = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if x.endswith(".jpg")]
    labels = data['Malignant'].values  # 假設 'label' 欄位存放良性或惡性標籤

    # 檢查並轉換非數值列（如有分類特徵）
    clinical_features = data.drop(columns=['Malignant'])  # 提取臨床特徵，去除 id 和標籤
    clinical_features_dim = clinical_features.shape[1]  # 記錄臨床特徵的維度

    # 檢查是否有非數值型別的列，並將它們轉換為數值
    for col in clinical_features.columns:
        if clinical_features[col].dtype == 'object':  # 如果是字串類型
            clinical_features[col] = pd.Categorical(clinical_features[col]).codes  # 將其轉換為分類數字
    # clinical_features = clinical_features.astype(np.float32)  # 將所有特徵轉換為 float32
    clinical_features = clinical_features.values  # 轉換為 numpy 陣列

    # 分割資料集
    X_train, X_test, y_train, y_test, train_clinical, test_clinical= train_test_split(image_paths, labels, clinical_features, test_size=test_size, random_state=random_seed, stratify=labels)
    print("Original train dataset size:", len(X_train))

    # 計算資料集的均值和標準差
    transform_for_stats = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 建立資料集
    train_dataset = PleuralEffusionDataset(X_train, y_train, train_clinical, transform=transform_for_stats)
    test_dataset = PleuralEffusionDataset(X_test, y_test, test_clinical, transform=transform_for_stats)

    return train_dataset, test_dataset, clinical_features_dim