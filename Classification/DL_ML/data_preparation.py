# data_preparation.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import logging
from sklearn.preprocessing import OneHotEncoder
import random
from sklearn.model_selection import train_test_split

from Classification.DL_ML.logger_utils import setup_log
from Classification.DL_ML.config import ConfigManager


class PleuralEffusionDataset(Dataset):
    """
    此 Dataset 會讀取影像並同時讀取臨床資料（來自 CSV 檔）。
    假設 CSV 檔中包含欄位： filename, label, 以及其他臨床特徵。
    """
    def __init__(self, image_paths, clinical_features, labels,  transform=None, augmented_images=None, augmented_clinical_features=None, augmented_labels=None):
        """
        :param image_paths: list，每個元素為完整的影像路徑。
        :param labels: list，每個元素為對應的分類標籤 (例如 0 或 1)。
        :param clinical_data: 若有，為一個 np.array 或 list，每筆臨床資料的數值向量。
        :param transform: torchvision transforms，對影像做前處理。
        """
        self.image_paths = image_paths
        self.labels = labels
        self.clinical_features = clinical_features  # 可為 None
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
            # 擴增影像
            image = self.augmented_images[idx - len(self.image_paths)]
            label = self.augmented_labels[idx - len(self.image_paths)]
            clinical_feature = self.clinical_features[idx - len(self.image_paths)]

        clinical_feature = torch.tensor(clinical_feature, dtype=torch.float32)  # 將臨床數據轉為 tensor   

        if self.transform:
            image = self.transform(image)
            # print(f"[Debug] Image {os.path.basename(image_path)[2:4]} range: min={image.min().item():.4f}, max={image.max().item():.4f}, mean={image.mean().item():.4f}")

        
        return image, clinical_feature, label, os.path.basename(image_path) if idx < len(self.image_paths) else f"augmented_{idx}"


class ClinicalDataPreprocessor:
    def __init__(self, config_manager: ConfigManager):
        self.data = pd.read_csv(config_manager.CSV_FILE)
        self.feature_type = config_manager.feature_type
        self.drop_columns = config_manager.drop_columns
        self.ohe_columns = ["Appearance", "Color"]
        self.encoder = OneHotEncoder(sparse_output=False)
        setup_log(config_manager.LOG_PATH)

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

def augment_data(image_paths, clinical_features, labels, augment_factor=4):
    augmented_images = []
    augmented_clinical_features = []
    augmented_labels = []

    for path, clinical_feature, label in zip(image_paths, clinical_features, labels):
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
                augmented_clinical_features.append(clinical_feature)
                augmented_labels.append(label)
    return augmented_images, augmented_clinical_features, augmented_labels

def prepare_data(config_manager: ConfigManager, test_size=0.2, random_seed=42):
    """
    讀取 CSV 檔後，根據圖片檔名組合完整路徑，並切分資料成訓練與測試集。
    CSV 應包含欄位 "filename"、"label" 及其他臨床特徵欄位。
    """    
    data_preprocessor = ClinicalDataPreprocessor(config_manager)
    data = data_preprocessor.preprocess_data()

    image_paths = [os.path.join(config_manager.IMAGE_FOLDER, x) for x in os.listdir(config_manager.IMAGE_FOLDER) if x.endswith(".jpg")]
    labels = data['Malignant'].values
   
    clinical_features = data.drop(columns=['Malignant'])  # 提取臨床特徵，去除 id 和標籤
    # 檢查是否有非數值型別的列，並將它們轉換為數值
    for col in clinical_features.columns:
        if clinical_features[col].dtype == 'object':  # 如果是字串類型
            clinical_features[col] = pd.Categorical(clinical_features[col]).codes  # 將其轉換為分類數字
    clinical_features = clinical_features.values  # 轉換為 numpy 陣列

    X_train, X_test, y_train, y_test, train_clinical, test_clinical= train_test_split(
        image_paths, labels, clinical_features, test_size=test_size, random_state=random_seed, stratify=labels)

    # 計算資料集的均值和標準差
    transform_for_stats = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 建立資料集
    train_dataset = PleuralEffusionDataset(X_train, train_clinical, y_train, transform=transform_for_stats)
    test_dataset = PleuralEffusionDataset(X_test, test_clinical, y_test, transform=transform_for_stats)

    # total = len(image_paths)
    # test_count = int(total * test_size)
    # train_count = total - test_count

    # train_image_paths = image_paths[:train_count]
    # train_labels = labels[:train_count]
    # train_clinical = clinical_features[:train_count]

    # test_image_paths = image_paths[train_count:]
    # test_labels = labels[train_count:]
    # test_clinical = clinical_features[train_count:]

    # train_data = (train_image_paths, train_labels, train_clinical)
    # test_data = (test_image_paths, test_labels, test_clinical)

    return train_dataset, test_dataset, data_preprocessor
