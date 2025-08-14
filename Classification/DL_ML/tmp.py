import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import yaml
import pandas as pd
import logging

from .data_preparation import PleuralEffusionDataset, ClinicalDataPreprocessor, prepare_data, augment_data
from .model import PleuralEffusionClassifier
from .logger_utils import setup_log
from .config import ConfigManager
from .focal_loss import FocalLoss
from .plotter_to_df import Plotter, SHAPPlotter, ReportToDataFrame

class DL_MLTraining:
    def __init__(self, config_manager: ConfigManager):
        self.cfg = config_manager
        self.RESULTS_PATH = os.path.join(self.cfg.FILE_PATH, 'results')
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        self.LOG_PATH = os.path.join(self.RESULTS_PATH, 'train.log')

    def run_kfold_training(self, num_folds=5):
        logging.info(f"Run k-fold training with {num_folds} folds.")

        full_train_dataset, self.test_dataset, data_preprocessor= prepare_data(self.cfg, test_size=0.2, random_seed=42)

        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        fold_num = []
        fold_f1 = []
        fold_accuracy = []
        fold_precision = []
        fold_recall = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(full_train_dataset)), full_train_dataset.labels)):
            logging.info(f"Starting fold {fold+1}/{num_folds}")

            train_image_paths = [full_train_dataset.image_paths[x] for x in train_idx]
            train_labels = [full_train_dataset.labels[x] for x in train_idx]
            train_clinical_features = [full_train_dataset.clinical_features[x] for x in train_idx]
            val_image_paths = [full_train_dataset.image_paths[x] for x in val_idx]
            val_labels = [full_train_dataset.labels[x] for x in val_idx]
            val_clinical_features = [full_train_dataset.clinical_features[x] for x in val_idx]

            extractor_trainer = TrainFeatureExtractor(self.cfg, fold)
            # train_combined_features, extractor_pth, pca = extractor_trainer.train_extractor(train_image_paths, train_clinical_features, train_labels, val_image_paths, val_clinical_features, val_labels)
            train_deep_feature, train_clinical, pca, extractor_pth = extractor_trainer.train_extractor(train_image_paths, train_clinical_features, train_labels, val_image_paths, val_clinical_features, val_labels)
            pca.fit(train_deep_feature)
            reduced_train_features = pca.transform(train_deep_feature)
            logging.info(f"after PCA, dim of reduced_train_features features: {reduced_train_features.shape[1]}")

            train_combined_features = np.concatenate([reduced_train_features, train_clinical], axis=1)
            logging.info(f"dim of train_combined_features: {train_combined_features.shape[1]}")

            # get deep features for validation set
            model, transform_normalize, device = self._load_checkpoint(extractor_pth)
            val_dataset = PleuralEffusionDataset(
                val_image_paths,                
                val_clinical_features, 
                val_labels,
                transform=transform_normalize
            )
            val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=False)
            val_deep_features, val_clinical_features, _ = extractor_trainer.extract_features(model, val_loader, device, all_train_amount=1) # all_train_amount 隨便設，是用於pca的，但這裡不需要回傳的pca
            reduced_val_features = pca.transform(val_deep_features)
            logging.info(f"after PCA, dim of reduced_val_features features: {reduced_val_features.shape[1]}")

            val_combined_features = np.concatenate([reduced_val_features, val_clinical_features], axis=1)
            logging.info(f"dim of val_combined_features: {val_combined_features.shape[1]}")
            
            ml = MachineLearning(self.cfg, fold, data_preprocessor)
            ml.train_all_model(train_combined_features, train_labels, val_combined_features, val_labels)

    def _load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
        checkpoint = torch.load(checkpoint_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PleuralEffusionClassifier(self.cfg.model_name, num_classes=2)
        model.to(device)
        model.load_state_dict(checkpoint['state_dict'])
        transform_normalize = checkpoint.get("normalization")
        logging.info(f"Normalization with {transform_normalize}")
        return model, transform_normalize, device

class TrainFeatureExtractor:
    def __init__(self, config_manager: ConfigManager, fold):
        self.cfg = config_manager
        self.fold = fold
        self.path_folder = os.path.join(self.cfg.RESULTS_FOLDER, 'pth')
        os.makedirs(self.path_folder, exist_ok=True)
        self.pth_path = os.path.join(self.path_folder, f"fold_{self.fold}_checkpoint.pth")

    def train_extractor(self, train_image_paths, train_clinical_features, train_labels, val_image_paths, val_clinical_features, val_labels):

        all_train_amount = len(train_image_paths)

        augmented_images, augmented_labels, augmented_clinical_features = [], [], []
        if self.cfg.aug:
            augmented_images, augmented_clinical_features, augmented_labels = augment_data(train_image_paths, train_clinical_features, train_labels)
            logging.info(
                f"Original amount of training images: {len(train_image_paths)}\t"
                f"Amount of validation iamges: {len(val_image_paths)}\t"
                f"Augmented images: {len(augmented_images)}\t"
                f"Total training images: {all_train_amount}"
            )
            all_train_amount += len(augmented_images)

        mean, std = self._compute_mean_std(train_image_paths, train_labels, train_clinical_features)
        logging.info(f"Fold {self.fold+1} mean: {mean}, std: {std}")

        transform_normalize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist())
        ])

        # Process augmentation
        if self.cfg.aug:
            train_dataset = PleuralEffusionDataset(
                train_image_paths,
                train_clinical_features,
                train_labels,
                transform=transform_normalize,
                augmented_images=augmented_images,
                augmented_clinical_features=augmented_clinical_features,
                augmented_labels=augmented_labels
            )
        else:
            train_dataset = PleuralEffusionDataset(
                train_image_paths,                    
                train_clinical_features, 
                train_labels,
                transform=transform_normalize
            )

        val_dataset = PleuralEffusionDataset(
            val_image_paths,                
            val_clinical_features, 
            val_labels,
            transform=transform_normalize
        )#

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PleuralEffusionClassifier(self.cfg.model_name, num_classes=2)
        model.to(device)

        criterion = self._get_loss_function()
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.learning_rate)
        optimizer_name = optimizer.__class__.__name__
        logging.info(f"Optimizer: {optimizer_name}")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        scheduler_name = scheduler.__class__.__name__
        logging.info(f"Scheduler: {scheduler_name}\t Step Size: {scheduler.step_size}\t Gamma: {scheduler.gamma}")

        early_stopping = EarlyStopping(path=self.pth_path, transform_normalize=transform_normalize, patience=5, verbose=True)
        
        for epoch in range(self.cfg.epochs):
            train_loss, train_acc = self._train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_labels, val_preds = self._validate(model, val_loader, criterion, device)
            print(
                f"Fold {self.fold+1}, Epoch {epoch+1}/{self.cfg.epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            logging.info(
                f"Epoch [{epoch+1}/{self.cfg.epochs}], "
                f"Train Loss: {train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
            )

            if self.cfg.early_stop:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered!")
                    logging.info("Early stopping triggered!")
                    break
            
        full_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=False)


        features_array, clinical_array, pca = self.extract_features(model, full_loader, device, all_train_amount)

        # pca = PCA(n_components=all_train_amount)
        # reduced_features = pca.fit_transform(features)
        # logging.info(f"after PCA, dim of traini_reduced_deep features: {reduced_features.shape[1]}")

        # combined_features = np.concatenate([reduced_features, clinical_features], axis=1)
        # logging.info(f"dim of train_combined_features: {combined_features.shape[1]}")
        return features_array, clinical_array, pca, self.pth_path
        
    def extract_features(self, model, dataloader, device, all_train_amount):
        """
        利用訓練好的模型進行推論，取得每筆影像的深度特徵。
        """
        logging.info("Extracting deep features...")
        model.eval()
        features_list = []
        labels_list = []
        clinical_list = []
        with torch.no_grad():
            for images, clinical, labels, _ in dataloader:
                images = images.to(device)
                _, features = model(images)  # get deep features
                features_list.append(features.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
                clinical_list.append(clinical.numpy())
        features_array = np.concatenate(features_list, axis=0)
        labels_array = np.array(labels_list)
        clinical_array = np.concatenate(clinical_list, axis=0)
        logging.info(f"dim of deep features: {features_array.shape[1]}\tdim of clinical features: {clinical_array.shape[1]}")

        pca = PCA(n_components=all_train_amount)
        # reduced_features = pca.fit_transform(features)
        pca.fit(features_array)

        # logging.info(f"after PCA, dim of traini_reduced_deep features: {reduced_features.shape[1]}")

        # combined_features = np.concatenate([reduced_features, clinical_array], axis=1)
        # logging.info(f"dim of train_combined_features: {combined_features.shape[1]}")
        return features_array, clinical_array, pca

    def _get_loss_function(self):
    
        if self.cfg.loss_function == 'weighted-ce':
            weight = torch.tensor([
                self.cfg.num_sample / self.cfg.num_benign,
                self.cfg.num_sample / self.cfg.num_malignant
            ])  
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weight = weight.to(device)
            criterion = nn.CrossEntropyLoss(weight=weight)
            print("Using Weighted Cross-Entropy Loss")
            return criterion

        elif self.cfg.loss_function == 'focal':
            focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
            print("Using Focal Loss")
            return focal_loss
        
        else:
            print("Using standard Cross-Entropy Loss")
            return nn.CrossEntropyLoss()
        
    def _compute_mean_std(self, image_paths, labels, clinical_features):
        
        transform_for_stats = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        dataset_for_stats = PleuralEffusionDataset(
            image_paths=image_paths,
            labels=labels,
            clinical_features=clinical_features,
            transform=transform_for_stats
        )
        loader_for_stats = DataLoader(dataset_for_stats, batch_size=self.cfg.batch_size, shuffle=False)

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
        return mean, std

    def _train_one_epoch(selg, model, dataloader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, _, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return running_loss/len(dataloader), correct/total  # return train loss and accuraty

    def _validate(self, model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        true_labels = []
        all_preds = []
        with torch.no_grad():
            for images, _, labels, _ in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                true_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
  
        return running_loss/len(dataloader), correct/total, true_labels, all_preds   # return val loss, val accuracy, labels, preds

    

# class ConcatenatedFeatureExtractor:
#     def __init__(self, config_manager: ConfigManager):
#         self.cfg = config_manager
#         self.RESULTS_PATH = os.path.join(self.cfg.FILE_PATH, 'results')
#         os.makedirs(self.RESULTS_PATH, exist_ok=True)
#         self.LOG_PATH = os.path.join(self.RESULTS_PATH, 'get_deep_features.log')
        
#     def run_kfold_training(self, dataset, num_folds=5):
#         """
#         對訓練資料使用 k-fold cross validation，
#         並在每個 fold 訓練後抽取深度特徵，接著與臨床資料 concat 後用傳統 ML (例如 SVM) 做分類。
#         """
#         logging.info(f"Run k-fold training with {num_folds} folds to extract deep features.")

#         full_train_dataset, self.test_dataset = prepare_data(self.cfg, test_size=0.2, random_seed=42)

#         # labels_np = np.array(train_dataset.labels)
#         print(f"train_dataset.labels: {full_train_dataset.labels}")
#         skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

#         fold_num = []
#         fold_f1 = []
#         fold_accuracy = []
#         fold_precision = []
#         fold_recall = []
#         for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(full_train_dataset)), full_train_dataset.labels)):
#             logging.info(f"Starting fold {fold+1}/{num_folds}")

#             train_image_paths = [full_train_dataset.image_paths[x] for x in train_idx]
#             train_labels = [full_train_dataset.labels[x] for x in train_idx]
#             train_clinical_features = [full_train_dataset.clinical_features[x] for x in train_idx]
#             val_image_paths = [full_train_dataset.image_paths[x] for x in val_idx]
#             val_labels = [full_train_dataset.labels[x] for x in val_idx]
#             val_clinical_features = [full_train_dataset.clinical_features[x] for x in val_idx]

#             all_train_amount = len(train_image_paths)

#             augmented_images, augmented_labels, augmented_clinical_features = [], [], []
#             if self.cfg.aug:
#                 augmented_images, augmented_clinical_features, augmented_labels = augment_data(train_image_paths, train_clinical_features, train_labels)
#                 logging.info(
#                     f"Original amount of training images: {len(train_image_paths)}\t"
#                     f"Amount of validation iamges: {len(val_image_paths)}\t"
#                     f"Augmented images: {len(augmented_images)}\t"
#                     f"Total training images: {all_train_amount}"
#                 )
#                 all_train_amount += len(augmented_images)

#             mean, std = self._compute_mean_std(train_image_paths, train_labels, train_clinical_features)
#             logging.info(f"Fold {fold+1} mean: {mean}, std: {std}")

#             transform_normalize = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=mean.tolist(), std=std.tolist())
#             ])

#             # Process augmentation
#             if self.cfg.aug:
#                 train_dataset = PleuralEffusionDataset(
#                     train_image_paths,
#                     train_clinical_features,
#                     train_labels,
#                     transform=transform_normalize,
#                     augmented_images=augmented_images,
#                     augmented_clinical_features=augmented_clinical_features,
#                     augmented_labels=augmented_labels
#                 )
#             else:
#                 train_dataset = PleuralEffusionDataset(
#                     train_image_paths,                    
#                     train_clinical_features, 
#                     train_labels,
#                     transform=transform_normalize
#                 )

#             val_dataset = PleuralEffusionDataset(
#                 val_image_paths,                
#                 val_clinical_features, 
#                 val_labels,
#                 transform=transform_normalize
#             )#

#             train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
#             val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=False)

#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             model = PleuralEffusionClassifier(self.cfg.model_name, num_classes=2)
#             model.to(device)

#             criterion = self._get_loss_function()
#             optimizer = optim.Adam(model.parameters(), lr=self.cfg.learning_rate)
#             optimizer_name = optimizer.__class__.__name__
#             logging.info(f"Optimizer: {optimizer_name}")
#             scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#             scheduler_name = scheduler.__class__.__name__
#             logging.info(f"Scheduler: {scheduler_name}\t Step Size: {scheduler.step_size}\t Gamma: {scheduler.gamma}")

#             train_losses = []
#             val_losses = []
#             train_accuracies = []
#             val_accuracies = []

#             early_stopping = EarlyStopping(patience=5, verbose=True, path=os.path.join(self.RESULTS_PATH, 'pth', f"fold_{fold}_checkpoint.pth"))
            
#             for epoch in range(self.cfg.epochs):
#                 train_loss, train_acc = self._train_one_epoch(model, train_loader, criterion, optimizer, device)
#                 val_loss, val_acc, val_labels, val_preds = self._validate(model, val_loader, criterion, device)
#                 print(
#                     f"Fold {fold+1}, Epoch {epoch+1}/{self.cfg.epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
#                 logging.info(
#                     f"Epoch [{epoch+1}/{self.cfg.epochs}], "
#                     f"Train Loss: {train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
#                 )

#                 if self.cfg.early_stop:
#                     early_stopping(val_loss, model)
#                     if early_stopping.early_stop:
#                         print("Early stopping triggered!")
#                         logging.info("Early stopping triggered!")
#                         break
                
#             full_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=False)
#             features, feature_labels, clinical_features = self._extract_features(model, full_loader, device)

#             pca = PCA(n_components=all_train_amount)
#             reduced_features = pca.fit_transform(features)
#             logging.info(f"after PCA, dim of reduced_deep features: {reduced_features.shape[1]}")

#             combined_features = np.concatenate([reduced_features, clinical_features], axis=1)
#             logging.info(f"dim of combined_features: {combined_features.shape[1]}")
    
#     def _get_loss_function(self):
    
#         if self.cfg.loss_function == 'weighted-ce':
#             weight = torch.tensor([
#                 self.cfg.num_sample / self.cfg.num_benign,
#                 self.cfg.num_sample / self.cfg.num_malignant
#             ])  
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             weight = weight.to(device)
#             criterion = nn.CrossEntropyLoss(weight=weight)
#             print("Using Weighted Cross-Entropy Loss")
#             return criterion

#         elif self.cfg.loss_function == 'focal':
#             focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
#             print("Using Focal Loss")
#             return focal_loss
        
#         else:
#             print("Using standard Cross-Entropy Loss")
#             return nn.CrossEntropyLoss()
        
#     def _compute_mean_std(self, image_paths, labels, clinical_features):
        
#         transform_for_stats = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor()
#         ])
#         dataset_for_stats = PleuralEffusionDataset(
#             image_paths=image_paths,
#             labels=labels,
#             clinical_features=clinical_features,
#             transform=transform_for_stats
#         )
#         loader_for_stats = DataLoader(dataset_for_stats, batch_size=self.cfg.batch_size, shuffle=False)

#         mean = torch.zeros(3)
#         std = torch.zeros(3)
#         total_images = 0
#         for images, *_ in loader_for_stats:
#             batch_samples = images.size(0)
#             total_images += batch_samples
#             mean += images.mean([0, 2, 3]) * batch_samples
#             std += images.std([0, 2, 3]) * batch_samples

#         mean /= total_images
#         std /= total_images
#         print(f"Mean: {mean}, Std: {std}")
#         return mean, std

#     def _train_one_epoch(selg, model, dataloader, criterion, optimizer, device):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#         for images, _, labels, _ in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs, _ = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         return running_loss/len(dataloader), correct/total  # return train loss and accuraty

#     def _validate(self, model, dataloader, criterion, device):
#         model.eval()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#         true_labels = []
#         all_preds = []
#         with torch.no_grad():
#             for images, _, labels, _ in dataloader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs, _ = model(images)
#                 loss = criterion(outputs, labels)
#                 running_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 true_labels.extend(labels.cpu().numpy())
#                 all_preds.extend(predicted.cpu().numpy())
  
#         return running_loss/len(dataloader), correct/total, true_labels, all_preds   # return val loss, val accuracy, labels, preds

#     def _extract_features(self, model, dataloader, device):
#         """
#         利用訓練好的模型進行推論，取得每筆影像的深度特徵。
#         """
#         logging.info("Extracting deep features...")
#         model.eval()
#         features_list = []
#         labels_list = []
#         clinical_list = []
#         with torch.no_grad():
#             for images, clinical, labels, _ in dataloader:
#                 images = images.to(device)
#                 _, features = model(images)  # get deep features
#                 features_list.append(features.cpu().numpy())
#                 labels_list.extend(labels.cpu().numpy())
#                 clinical_list.append(clinical.numpy())
#         features_array = np.concatenate(features_list, axis=0)
#         labels_array = np.array(labels_list)
#         clinical_array = np.concatenate(clinical_list, axis=0)
#         logging.info(f"dim of deep features: {features_array.shape[1]}\tdim of clinical features: {clinical_array.shape[1]}")

#         return features_array, labels_array, clinical_array

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, path, transform_normalize, patience=5, verbose=False, delta=0, ):
        """
        :param patience: 沒有改善的 epoch 數量
        :param verbose: 是否打印訊息
        :param delta: 最小的改善量
        :param path: 儲存 checkpoint 的檔案路徑
        """
        self.path = path
        self.transform_normalize = transform_normalize
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss  # 因為 loss 越低越好
        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self._save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "normalization": self.transform_normalize, 
                }, val_loss
            )
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "normalization": self.transform_normalize, 
                }, val_loss
            )            
            self.counter = 0

    def _save_checkpoint(self, state, val_loss):
        '''當驗證 loss 改善時，儲存模型'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(state, self.path)
        logging.info(f"Model saved at {self.path}")
        self.val_loss_min = val_loss

class MachineLearning:
    def __init__(self, config_manager: ConfigManager, fold_index, data_preprocessor: ClinicalDataPreprocessor):
        self.cfg = config_manager
        self.fold_index = fold_index
        self.dp = data_preprocessor
        
    def train_all_model(self, x_train, y_train, x_test, y_test):
        model_names = self.cfg['ML_model']['name']['options']
        model_metrics = {
            model_name: {
                'f1': [],
                'accuracy': [],
                'precision': [],
                'recall': []
            }
            for model_name in model_names
        }
        for model_name in model_names:
            best_model, model = self._gridsearch_train_one_model(x_train, y_train, model_name)   
            f1, acc, precision, recall = self._evaluate_model(
                x_train, x_test, y_test, best_model, model_name, self.fold_index
            )

            model_metrics[model_name]['f1'].append(f1)
            model_metrics[model_name]['accuracy'].append(acc)
            model_metrics[model_name]['precision'].append(precision)
            model_metrics[model_name]['recall'].append(recall)

        logging.info(f"Fold {self.fold_index} all models trained and evaluated complete.")
        
        for model_name in model_names:
            f1_mean = np.mean(model_metrics[model_name]['f1'])
            f1_std  = np.std(model_metrics[model_name]['f1'])
            acc_mean = np.mean(model_metrics[model_name]['accuracy'])
            acc_std  = np.std(model_metrics[model_name]['accuracy'])
            prec_mean = np.mean(model_metrics[model_name]['precision'])
            prec_std  = np.std(model_metrics[model_name]['precision'])
            rec_mean = np.mean(model_metrics[model_name]['recall'])
            rec_std  = np.std(model_metrics[model_name]['recall'])

            logging.info(f"Model: {model_name}")
            logging.info(f"  F1 Score     : mean={f1_mean:.4f}, std={f1_std:.4f}")
            logging.info(f"  Accuracy     : mean={acc_mean:.4f}, std={acc_std:.4f}")
            logging.info(f"  Precision    : mean={prec_mean:.4f}, std={prec_std:.4f}")
            logging.info(f"  Recall       : mean={rec_mean:.4f}, std={rec_std:.4f}")


    def _gridsearch_train_one_model(self, x_train, y_train, model_name):
        self.now_model, self.param_grid = self._get_model_and_params(model_name)
        logging.info(f"Grid Search for {model_name} started.")
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'precision_macro': make_scorer(precision_score, average='macro'),
            'recall_macro': make_scorer(recall_score, average='macro'),
        }
        grid_search = GridSearchCV(
            self.now_model, 
            self.param_grid, 
            scoring=scoring,
            refit='f1_macro',
            cv=5, 
            n_jobs=-1, 
            verbose=1
        )
        grid_search.fit(x_train, y_train)

        cv_results = pd.DataFrame(grid_search.cv_results_)
        columns_to_log = [col for col in cv_results.columns if 'mean_test' in col or 'std_test' in col]
        cv_results[columns_to_log].to_csv(f'{self.cfg.RESULTS_FOLDER}/{model_name}_cv_results.csv')

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        best_index = grid_search.best_index_
        best_metrics = cv_results.loc[best_index, columns_to_log]
        logging.info(f"Best Parameters: {best_params}\nBest metrics: \n{best_metrics}")
        logging.info(f"--- {model_name} model trained successfully. ---\n")
        
        return best_model, self.now_model
    
    def _evaluate_model(self, x_train, x_test, y_test, best_model, model_name, fold_index):
        logging.info(f"--- Evaluating {model_name} model started. ---")
        y_pred = best_model.predict(x_test)
        y_pred_probs = best_model.predict_proba(x_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Testing Accuracy: {accuracy}")
        Plotter.plot_confusion_matrix(y_test, y_pred, self.cfg.confusion_matrix_output_dir, model_name, fold_index)
        Plotter.plot_roc_curve(y_test, y_pred_probs, self.cfg.ROC_output_dir, model_name, fold_index)
        SHAPlotter = SHAPPlotter(self.dp, x_train, best_model.predict, self.cfg.shap_output_dir, model_name, fold_index)
        SHAPlotter.plot_shape_values()
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = ReportToDataFrame._classification_report_to_df(report)
        logging.info(f"Classification Report: \n{report_df}")
        logging.info(f"--- {model_name} model evaluated successfully. ---\n")
        f1 = report['weighted avg']['f1-score']
        acc = accuracy
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        return f1, acc, precision, recall

    def _get_model_and_params(self, model_name=None):
        logging.info(f"{'#'*30}")
        logging.info(f"Using model: {model_name}")
        model_params = self.cfg['ML_model']['parameters'][model_name]
        if model_name == 'DecisionTree':
            model = DecisionTreeClassifier(class_weight='balanced')
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(class_weight='balanced')
        elif model_name == 'SVM':
            model = SVC(class_weight='balanced', probability=True)
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(class_weight='balanced', penalty='l2')
        elif model_name == 'XGBoost':
            model = XGBClassifier(eval_metric='logloss', scale_pos_weight=3.35)     
            # scale_pos_weight = Benign(negative)/Malignant(positive) ratio = 77/23 = 3.35
        
        param_grid = {key: value['options'] for key, value in model_params.items()}
        logging.info(f"Model Parameters combination as below: \n{param_grid}")
        return model, param_grid

    def _scale_feature(self, x_train, x_test):
        if self.cfg.scale_way == 'StandardScaler':
            scaler = StandardScaler()
            logging.info(f"Training feature scaled using StandardScaler\n\n")
        elif self.cfg.scale_way == 'MinMaxScaler':
            scaler = MinMaxScaler()
            logging.info(f"Training feature scaled using MinMaxScaler\n\n")
        elif self.cfg.scale_way == 'RobustScaler':
            scaler = RobustScaler()
            logging.info(f"Training feature scaled using RobustScaler\n\n")
        return scaler.fit_transform(x_train), scaler.transform(x_test)

def main():
    config_path = 'Classification/DL_ML/config.yaml'
    config_manager = ConfigManager(config_path)
    setup_log(config_manager.LOG_PATH)

    # 用於訓練深度學習網路，將所有資料分成 5-fold，不另外分出 test，每 fold 的 val set 用該 fold 提取深度特徵避免 data leakage
    # train_dataset = prepare_data(config_manager)
    
    # 使用 k-fold cross validation 在訓練資料上訓練深度網路並抽取特徵，再利用傳統 ML 分類
    # feature_extractor = ConcatenatedFeatureExtractor(config_manager)
    # fold_results = feature_extractor.run_kfold_training(train_dataset, num_folds=5)
    trainer = DL_MLTraining(config_manager)
    trainer.run_kfold_training(num_folds=5)
    # for res in fold_results:
    #     print(res)
    
    # 注意：若要最終部署，可在選定最佳超參數後用全部訓練資料重訓一個模型，
    # 然後用此模型對測試資料抽取特徵，接著再用相同流程進行下游分類。
    
if __name__ == "__main__":
    main()









# # train.py

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# from torchvision import transforms
# from sklearn.model_selection import StratifiedKFold
# from sklearn.decomposition import PCA
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np

# from data_preparation import PleuralEffusionDataset, prepare_data
# from model import PleuralEffusionClassifier

# def train_one_epoch(model, dataloader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for images, labels, clinical in dataloader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs, _ = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     return running_loss/len(dataloader), correct/total

# def validate(model, dataloader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     all_labels = []
#     all_preds = []
#     with torch.no_grad():
#         for images, labels, clinical in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs, _ = model(images)
#             loss = criterion(outputs, labels)
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(predicted.cpu().numpy())
#     return running_loss/len(dataloader), correct/total, all_labels, all_preds

# def extract_features(model, dataloader, device):
#     """
#     利用訓練好的模型進行推論，取得每筆影像的深度特徵。
#     """
#     model.eval()
#     features_list = []
#     labels_list = []
#     clinical_list = []
#     with torch.no_grad():
#         for images, labels, clinical in dataloader:
#             images = images.to(device)
#             _, features = model(images)  # 取得深度特徵
#             features_list.append(features.cpu().numpy())
#             labels_list.extend(labels.cpu().numpy())
#             clinical_list.append(clinical.numpy())
#     features_array = np.concatenate(features_list, axis=0)
#     labels_array = np.array(labels_list)
#     clinical_array = np.concatenate(clinical_list, axis=0)
#     return features_array, labels_array, clinical_array

# def run_kfold_training(train_data, num_folds=5, epochs=5, batch_size=16, lr=1e-4):
#     """
#     對訓練資料 (影像 + 臨床資料) 使用 k-fold cross validation，
#     並在每個 fold 訓練後抽取深度特徵，接著與臨床資料 concat 後用傳統 ML (例如 SVM) 做分類。
#     """
#     image_paths, labels, clinical = train_data
#     # 定義影像前處理：resize、ToTensor 與 normalization
#     transform_train = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
#     ])
#     dataset = PleuralEffusionDataset(image_paths, labels, clinical_data=clinical, transform=transform_train)
#     labels_np = np.array(labels)

#     skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
#     fold_results = []
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels_np)):
#         print(f"Starting fold {fold+1}/{num_folds}")
#         # 分別取出此 fold 的訓練集與驗證集
#         train_subset = Subset(dataset, train_idx)
#         val_subset = Subset(dataset, val_idx)
#         train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
#         # 建立並微調模型（以 resnet50 為例）
#         model = PleuralEffusionClassifier(model_name="resnet50", num_classes=2)
#         model = model.to(device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=lr)
        
#         for epoch in range(epochs):
#             train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
#             val_loss, val_acc, val_labels, val_preds = validate(model, val_loader, criterion, device)
#             print(f"Fold {fold+1}, Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
#         # 使用訓練完的模型對整個訓練資料（train+val）抽取深度特徵
#         full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#         features, feature_labels, clinical_features = extract_features(model, full_loader, device)
#         # 合併影像深度特徵與臨床資料 (假設 clinical_features 已為向量)
#         combined_features = np.concatenate([features, clinical_features], axis=1)
        
#         # 可選：使用 PCA 進行降維
#         # from sklearn.decomposition import PCA
#         # pca = PCA(n_components=100)
#         # reduced_features = pca.fit_transform(combined_features)
#         # 這裡我們直接使用 combined_features
        
#         # 下游機器學習分類器：以 SVM 為例
#         svm = SVC(kernel="rbf", C=1.0)
#         svm.fit(combined_features, feature_labels)
#         ml_preds = svm.predict(combined_features)
#         ml_accuracy = accuracy_score(feature_labels, ml_preds)
#         print(f"Fold {fold+1} downstream ML classifier accuracy: {ml_accuracy:.4f}")
        
#         fold_results.append({
#             "fold": fold+1,
#             "val_accuracy": val_acc,
#             "ml_accuracy": ml_accuracy,
#             "classification_report": classification_report(feature_labels, ml_preds, output_dict=True)
#         })
    
#     return fold_results

# def main():
#     # 設定影像資料夾與 CSV 路徑，請根據實際路徑修改
#     image_folder = "path/to/your/images"
#     csv_file = "path/to/your/clinical_feature.csv"
    
#     # 取得訓練與測試資料 (這裡先將資料分為 train 與 test)
#     train_data, test_data = prepare_data(image_folder, csv_file, test_split=0.2)
    
#     # 使用 k-fold cross validation 在訓練資料上訓練深度網路並抽取特徵，再利用傳統 ML 分類
#     fold_results = run_kfold_training(train_data, num_folds=5, epochs=5, batch_size=16, lr=1e-4)
#     print("K-fold training results:")
#     for res in fold_results:
#         print(res)
    
#     # 注意：若要最終部署，可在選定最佳超參數後用全部訓練資料重訓一個模型，
#     # 然後用此模型對測試資料抽取特徵，接著再用相同流程進行下游分類。
    
# if __name__ == "__main__":
#     main()




# out of fold for loop model_metrics:  
import pandas as pd


class MetricsToDatafram:
    @staticmethod

    def model_metrics_to_pretty_df(model_metrics: dict) -> pd.DataFrame:
        """
        將 model_metrics 一次轉換成整齊漂亮的 DataFrame：
        - 每個模型的 metrics 展開
        - 每個模型名稱只出現一次
        - 模型之間插入空行

        回傳:
            pd.DataFrame：適合列印或匯出用的美化後表格
        """
        formatted_rows = []

        for model_idx, (model_name, metrics) in enumerate(model_metrics.items()):
            num_folds = len(next(iter(metrics.values())))
            
            for metric_name, values in metrics.items():
                row = {'Model': model_name, 'Metric': metric_name}
                for i in range(num_folds):
                    row[f'fold {i+1}'] = round(values[i], 4)
                formatted_rows.append(row)
            
            # 插入空行（不是最後一個模型才插）
            if model_idx < len(model_metrics) - 1:
                empty_row = {col: '' for col in ['Model', 'Metric'] + [f'fold {i+1}' for i in range(num_folds)]}
                formatted_rows.append(empty_row)

        # 移除重複的模型名稱（只保留第一次）
        last_model = None
        for row in formatted_rows:
            if row['Model'] == last_model:
                row['Model'] = ''
            else:
                last_model = row['Model']

        return pd.DataFrame(formatted_rows)

    
def main():
    model_metrics = {'DecisionTree': 
    {
    'f1': [0.8854166666666666, 
            0.8188405797101449, 
            0.6428571428571428, 
            0.5909090909090909, 
            0.625], 
    'accuracy': [0.875, 
                 0.8125, 
                 0.75, 
                 0.5625, 
                 0.625], 
    'precision': [0.925, 
                  0.8318181818181818, 
                  0.5625, 
                  0.6547619047619048, 
                  0.625], 
    'recall': [0.875, 0.8125, 0.75, 0.5625, 0.625]}, 
    
    'RandomForest': {
        'f1': [0.5317460317460317, 0.75, 0.6428571428571428, 0.6454545454545454, 0.611111111111111], 
        'accuracy': [0.5, 0.75, 0.75, 0.625, 0.6875], 
        'precision': [0.8636363636363636, 0.75, 0.5625, 0.6833333333333335, 0.5499999999999999], 
        'recall': [0.5, 0.75, 0.75, 0.625, 0.6875]}, 
        
    'SVM': {
        'f1': [0.05921052631578947, 0.1, 0.6428571428571428, 0.1, 0.611111111111111], 
        'accuracy': [0.1875, 0.25, 0.75, 0.25, 0.6875], 
        'precision': [0.03515625, 0.0625, 0.5625, 0.0625, 0.5499999999999999], 
        'recall': [0.1875, 0.25, 0.75, 0.25, 0.6875]}, 

    'LogisticRegression': {
        'f1': [0.05921052631578947, 0.6428571428571428, 0.6428571428571428, 0.1, 0.611111111111111], 
        'accuracy': [0.1875, 0.75, 0.75, 0.25, 0.6875], 
        'precision': [0.03515625, 0.5625, 0.5625, 0.0625, 0.5499999999999999], 
        'recall': [0.1875, 0.75, 0.75, 0.25, 0.6875]}, 
        
    'XGBoost': {
        'f1': [0.6098484848484849, 0.5772946859903382, 0.576923076923077, 0.75, 0.625], 
        'accuracy': [0.5625, 0.5625, 0.625, 0.75, 0.625], 
        'precision': [0.7578125, 0.5954545454545455, 0.5357142857142857, 0.75, 0.625], 
        'recall': [0.5625, 0.5625, 0.625, 0.75, 0.625]}}
    
    df = MetricsToDatafram.model_metrics_to_pretty_df(model_metrics)
    print(df)

if __name__ == "__main__":
    main()


DT: {
                'f1': [],
                'accuracy': [],
                'precision': [],
                'recall': []
            }
RF: {
                'f1': [],
                'accuracy': [],
                'precision': [],
                'recall': []
            }
SVM: {
                'f1': [],
                'accuracy': [],
                'precision': [],
                'recall': []
            }
LR: {
                'f1': [],
                'accuracy': [],
                'precision': [],
                'recall': []
            }
XG: {
                'f1': [],
                'accuracy': [],
                'precision': [],
                'recall': []
            }
