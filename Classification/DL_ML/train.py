import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import pandas as pd
import logging
import argparse
import statistics
import matplotlib.pyplot as plt

from Classification.DL_ML.data_preparation import PleuralEffusionDataset, ClinicalDataPreprocessor, prepare_data, augment_data
from Classification.DL_ML.model import PleuralEffusionClassifier
from Classification.DL_ML.logger_utils import setup_log
from Classification.DL_ML.config import ConfigManager
from Classification.DL_ML.focal_loss import FocalLoss
from Classification.DL_ML.plotter_to_df import Plotter, ReportToDataFrame, MetricsToDatafram

class DL_MLTraining:
    def __init__(self, config_manager: ConfigManager):
        self.cfg = config_manager

    def run_kfold_training(self, num_folds=5):
        logging.info(f"Run k-fold training with {num_folds} folds.")

        full_train_dataset, self.test_dataset, data_preprocessor= prepare_data(self.cfg, test_size=0.2, random_seed=42)

        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

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
        extract_folder = r'Classification\DL\image_only\results\despeckle\resnet50\20250403_002801_focal_0.001_aug_False'
        for tfold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(full_train_dataset)), full_train_dataset.labels)):
            fold = tfold + 1
            logging.info(f"{'-'*20}Starting fold {fold}/{num_folds}{'-'*20}")
            logging.info(f"Verify that the data splitting is consistent across each execution:")
            logging.info(f"Fold {fold} - Train indices: {train_idx} | Val indices: {val_idx}\n\n")


            train_image_paths = [full_train_dataset.image_paths[x] for x in train_idx]
            train_labels = [full_train_dataset.labels[x] for x in train_idx]
            train_clinical_features = [full_train_dataset.clinical_features[x] for x in train_idx]
            val_image_paths = [full_train_dataset.image_paths[x] for x in val_idx]
            val_labels = [full_train_dataset.labels[x] for x in val_idx]
            val_clinical_features = [full_train_dataset.clinical_features[x] for x in val_idx]

            extractor_trainer = TrainFeatureExtractor(self.cfg, fold)
            # train_deep_feature, train_clinical, train_labels, pca, extractor_pth = extractor_trainer.train_extractor(train_image_paths, train_clinical_features, train_labels, val_image_paths, val_clinical_features, val_labels)
            # scaler_deep_feature = StandardScaler()
            # scaler_deep_feature.fit(train_deep_feature)
            # scaled_train_deep_feature = scaler_deep_feature.transform(train_deep_feature)

            # pca.fit(scaled_train_deep_feature)
            # print(f"PCA selected components: {pca.n_components_}")
            # logging.info(f"PCA selected components: {pca.n_components_}")
            # logging.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

            # reduced_train_features = pca.transform(scaled_train_deep_feature)            
            # plt.figure(figsize=(8, 6))
            # colors = ['blue', 'red']
            # class_names = ['Benign', 'Malignant']
            # # plt.scatter(reduced_train_features[:, 0], reduced_train_features[:, 1], c=train_labels, cmap='viridis', alpha=0.7)
            # for class_idx, class_name in enumerate(class_names):
            #     mask = (train_labels == class_idx)
            #     plt.scatter(
            #         reduced_train_features[mask, 0],
            #         reduced_train_features[mask, 1],
            #         c=colors[class_idx],
            #         label=class_name,      # 直接使用類別名稱做圖例
            #         alpha=0.7
            #     )
            # plt.xlabel("Principal Component 1")
            # plt.ylabel("Principal Component 2")
            # plt.title("PCA of Deep Features")
            # # plt.colorbar(label='Label')
            # plt.legend()
            # plt.savefig(self.cfg.RESULTS_FOLDER + f"/fold_{fold}_pca.png")
            # plt.close()

            scaler_clinical = StandardScaler()
            # scaler_clinical.fit(train_clinical)
            # scaled_train_clinical = scaler_clinical.transform(train_clinical)
            # train_combined_features = np.concatenate([scaled_train_clinical, reduced_train_features], axis=1)

            # logging.info(f"dim of train_combined_features: {train_combined_features.shape}")

            #@@@@@@@@@@
            extractor_pth = os.path.join(extract_folder, f"checkpoint_fold_{fold}.pth")
            logging.info(f"extractor path: {extractor_pth}")
            #@@@@@@@@@

            # get deep features for validation set
            # logging.info("Validation set deep feature extraction...")
            model, transform_normalize, device = self._load_checkpoint(extractor_pth)
            
            #@@@@@@@@@
            logging.info(f"extracting training set deep features...")
            train_dataset = PleuralEffusionDataset(
                train_image_paths,                
                train_clinical_features, 
                train_labels,
                transform=transform_normalize
            )
            train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=False)
            
            train_deep_features, train_clinical, _ = extractor_trainer.extract_features(model, train_loader, device)
            logging.info(f"dim of train_deep_features: {train_deep_features.shape[1]}\t\tdim of train clinical features: {train_clinical.shape[1]}")
            variances = np.var(train_deep_features, axis=0)
            logging.info(f"Variance of each feature dimension: {variances}")
            logging.info(f"Matrix rank: {np.linalg.matrix_rank(train_deep_features)}")
            
            scaler_deep_feature = StandardScaler()
            scaler_deep_feature.fit(train_deep_features)
            scaled_train_deep_features = scaler_deep_feature.transform(train_deep_features)
            variances = np.var(scaled_train_deep_features, axis=0)
            logging.info(f"Variance of each scaled feature dimension: {variances}")
            
            pca = PCA(n_components=0.95, svd_solver='full')
            pca.fit(scaled_train_deep_features)
            logging.info(f"PCA selected training set components: {pca.n_components_}")
            logging.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

            reduced_train_deep_features = pca.transform(scaled_train_deep_features)
            logging.info(f"after PCA, dim of reduced_train_deep_features features: {reduced_train_deep_features.shape}")
            print("PC1 range:", reduced_train_deep_features[:, 0].min(), reduced_train_deep_features[:, 0].max())
            print("PC2 range:", reduced_train_deep_features[:, 1].min(), reduced_train_deep_features[:, 1].max())

            Plotter.plot_pca(train_labels, reduced_train_deep_features, self.cfg.pca_output_dir + f"train_fold_{fold}_pca.png")

            scaler_clinical.fit(train_clinical)
            
            scaled_train_clinical = scaler_clinical.transform(train_clinical)
            train_combined_features = np.concatenate([scaled_train_clinical, reduced_train_deep_features], axis=1)

            logging.info(f"dim of train_combined_features: {train_combined_features.shape}\n\n")
            #@@@@@@@@
              
            logging.info(f"Extracting validation set deep features...")
            val_dataset = PleuralEffusionDataset(
                val_image_paths,                
                val_clinical_features, 
                val_labels,
                transform=transform_normalize
            )
            val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=False)
            
            val_deep_features, val_clinical, _ = extractor_trainer.extract_features(model, val_loader, device)
            logging.info(f"dim of val_deep_features: {val_deep_features.shape[1]}\t\tdim of clinical features: {val_clinical.shape[1]}")
            variances = np.var(val_deep_features, axis=0)
            logging.info(f"Variance of each feature dimension: {variances}")
            logging.info(f"Matrix rank: {np.linalg.matrix_rank(val_deep_features)}")
            scaled_val_deep_features = scaler_deep_feature.transform(val_deep_features)
            variances = np.var(scaled_val_deep_features, axis=0)
            logging.info(f"Variance of each scaled feature dimension: {variances}")
            
            reduced_val_deep_features = pca.transform(scaled_val_deep_features)
            logging.info(f"after PCA, dim of reduced_val_features features: {reduced_val_deep_features.shape[1]}")
            Plotter.plot_pca(val_labels, reduced_val_deep_features, self.cfg.pca_output_dir + f"val_fold_{fold}_pca.png")

            scaled_val_clinical = scaler_clinical.transform(val_clinical)

            val_combined_features = np.concatenate([scaled_val_clinical, reduced_val_deep_features], axis=1)
            logging.info(f"dim of val_combined_features: {val_combined_features.shape}\n\n")
            
            logging.info("training set and validation set features all ready, start ML training...")
            ml = MachineLearning(self.cfg, fold, data_preprocessor)
            # model_metrics = ml.train_all_model(train_combined_features, train_labels, val_combined_features, val_labels, model_metrics)
            model_metrics = ml.train_all_model(reduced_train_deep_features, train_labels, reduced_val_deep_features, val_labels, model_metrics)

        
        logging.info(f"5-Fold DL_ML training complete.\n\n")

        logging.info("5-fold validation set results:")
        logging.info("All model results summarize:")
        model_metrics_df = MetricsToDatafram._model_metrics_to_df(model_metrics)
        logging.info(f"\n{model_metrics_df}")
        logging.info("\n\n")
        logging.info("5-fold validation set results mean & std.")

        for model_name, metrics in model_metrics.items():
            logging.info(f'{model_name}:')
            for metric_name, values in metrics.items():
                mean = statistics.mean(values)
                std = statistics.stdev(values)
                logging.info(f"{metric_name:<12}:  mean = {mean:.4f}, std = {std:.4f}")
            logging.info("\n")
        logging.info(f"Training complete.")

    def _load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PleuralEffusionClassifier(self.cfg, num_classes=2)
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
            all_train_amount += len(augmented_images)

        logging.info(
                f"Original amount of training images: {len(train_image_paths)}\t"
                f"Amount of validation iamges: {len(val_image_paths)}\t"
                f"Augmented images: {len(augmented_images)}\t"
                f"Total training images: {all_train_amount}"
            )
        mean, std = self._compute_mean_std(train_image_paths, train_labels, train_clinical_features)
        logging.info(f"Fold {self.fold} mean: {mean}, std: {std}")

        transform_normalize = transforms.Compose([
            transforms.Resize((256, 256)),
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
        model = PleuralEffusionClassifier(self.cfg, num_classes=2)
        model.to(device)

        criterion = self._get_loss_function()
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.learning_rate)
        optimizer_name = optimizer.__class__.__name__
        logging.info(f"Optimizer: {optimizer_name}")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        scheduler_name = scheduler.__class__.__name__
        logging.info(f"Scheduler: {scheduler_name}\t Step Size: {scheduler.step_size}\t Gamma: {scheduler.gamma}")

        early_stopping = EarlyStopping(path=self.pth_path, transform_normalize=transform_normalize, patience=self.cfg.patience, verbose=True)
        
        for epoch in range(self.cfg.epochs):
            train_loss, train_acc = self._train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_labels, val_preds, filenames_list = self._validate(model, val_loader, criterion, device)
            print(
                f"Fold {self.fold}, Epoch {epoch+1}/{self.cfg.epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
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
        logging.info("Deep feature extractor training complete.\n\n")
        val_df = pd.DataFrame({
            'Filename': filenames_list,
            'True Label': val_labels,
            'Predicted Label': val_preds,
        })
        logging.info(f"Results of fold {self.fold} validation: \n{val_df}")
        logging.info("Deep learning classification report:")
        report_dict = classification_report(
                val_labels, val_preds,
                zero_division='warn',
                output_dict=True
        )
        report_df = ReportToDataFrame._classification_report_to_df(report_dict)
        logging.info(f"Classification Report: \n{report_df}")
        full_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=False)
        features_array, clinical_array, labels = self.extract_features(model, full_loader, device)

        return features_array, clinical_array, labels, self.pth_path
        
    def extract_features(self, model, dataloader, device, ncomponents=0.95):
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
        # logging.info(f"dim of deep features: {features_array.shape[1]}\t\tdim of clinical features: {clinical_array.shape[1]}")

        # pca = PCA(n_components=ncomponents, svd_solver='full')

        return features_array, clinical_array, labels_array

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

    def _train_one_epoch(self, model, dataloader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, _, labels, filenames in dataloader:
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
        filenames_list = []
        with torch.no_grad():
            for images, _, labels, filenames in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                true_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                filenames_list.extend(filenames)

  
        return running_loss/len(dataloader), correct/total, true_labels, all_preds, filenames_list   # return val loss, val accuracy, labels, preds

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, path, transform_normalize, patience, verbose=False, delta=0, ):
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
        
        
    def train_all_model(self, x_train, y_train, x_test, y_test, model_metrics):
        model_names = self.cfg['ML_model']['name']['options']

        for model_name in model_names:
            best_model, model = self._gridsearch_train_one_model(x_train, y_train, model_name)   
            f1, acc, precision, recall = self._evaluate_model(
                x_train, x_test, y_test, best_model, model_name, self.fold_index
            )

            model_metrics[model_name]['f1'].append(f1)
            model_metrics[model_name]['accuracy'].append(acc)
            model_metrics[model_name]['precision'].append(precision)
            model_metrics[model_name]['recall'].append(recall)

        return model_metrics

    def _gridsearch_train_one_model(self, x_train, y_train, model_name):
        self.now_model, self.param_grid = self._get_model_and_params(model_name)
        print(f"Grid Search for {model_name} started.")
        logging.info(f"Grid Search for {model_name} started.")
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
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
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = ReportToDataFrame._classification_report_to_df(report)
        logging.info(f"Classification Report: \n{report_df}")
        logging.info(f"--- {model_name} model evaluated successfully. ---\n")
        f1 = report['weighted avg']['f1-score']
        acc = accuracy
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        return f1, acc, precision, recall

    def _get_model_and_params(self, model_name=None):
        logging.info(f"{'-'*50}")
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

    # def _scale_feature(self, x_train, x_test):
    #     if self.cfg.scale_way == 'StandardScaler':
    #         scaler = StandardScaler()
    #         logging.info(f"Training feature scaled using StandardScaler")
    #     elif self.cfg.scale_way == 'MinMaxScaler':
    #         scaler = MinMaxScaler()
    #         logging.info(f"Training feature scaled using MinMaxScaler")
    #     elif self.cfg.scale_way == 'RobustScaler':
    #         scaler = RobustScaler()
    #         logging.info(f"Training feature scaled using RobustScaler")
    #     return scaler.fit_transform(x_train), scaler.transform(x_test)

def main():
    # config_path = 'Classification/DL_ML/config.yaml'
    # config_manager = ConfigManager(config_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = args.config
    config_manager = ConfigManager(config_path)
    setup_log(config_manager.LOG_PATH)

    trainer = DL_MLTraining(config_manager)
    trainer.run_kfold_training(num_folds=5)
 
    
if __name__ == "__main__":
    main()
