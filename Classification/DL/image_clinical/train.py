import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import itertools
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold


from .data_preparation import prepare_data, augment_data, PleuralEffusionDataset,ClinicalDataPreprocessor
from .model import PleuralEffusionClassifier


class ConfigManager:
    
    def __init__(self, config_path=None, config=None):
        self._config = None
        if config is not None:
            self._config = config
        elif config_path is not None:
            self.config_path = config_path
            self._config = self._load_config()
        else:
            raise ValueError("Either config_path or config must be provided")

        self.num_folds = self._config['num_folds']
        self.model_name = self._config['model_parameter']['model_name']['default']
        self.learning_rate = self._config['model_parameter']['learning_rate']['default']
        self.loss_function = self._config['model_parameter']['loss_function']['default']
        self.epochs = self._config['model_parameter']['epochs']
        self.batch_size = self._config['model_parameter']['batch_size']
        self.drop_columns = self._config['data_setting']['drop_columns']

        self.image_type = self._config['data_setting']['image_type']['default']
        self.feature_type = self._config['data_setting']['feature_type']['default']
        self.aug = self._config['data_setting']['aug']['default']
        self.num_sample = self._config['data_setting']['num_sample']
        self.num_malignant = self._config['data_setting']['num_malignant']
        self.num_benign = self.num_sample - self.num_malignant

        self.current_datetime = pd.Timestamp.now()
        self.timestamp = self.current_datetime.strftime("%Y%m%d_%H%M%S")
        self.FILE_PATH = 'Classification/DL/image_clinical'
        self.IMAGE_FOLDER = self._determine_image_folder()
        self.CSV_FILE = os.path.join('Classification', f"{self._config['data_setting']['feature_type']['default']}.csv")

    def _load_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _determine_image_folder(self):
        if self.image_type == 'original':
            return 'Preprocessing/Cropped'
        elif self.image_type == 'CLAHE':
            return 'Preprocessing/CLAHE'
        elif self.image_type == 'despeckle':
            return 'Preprocessing/CLAHE_BM3D'
        elif self.image_type == 'UNetPlusPlus-roi':
            return 'Segmentation/UNetPlusPlus/results/all_roi'
        elif self.image_type == 'GLFR-roi':
            return 'Segmentation/GLFR-main/results/all_roi'
        elif self.image_type == 'x-ray':
            return r'C:\Users\ashle\Ashlee\BusLab\workkkk\RawDataset\X-Ray'
        else:
            raise ValueError(f"Unknown image_type: {self.image_type}")

class Plotter:
    @staticmethod
    def plot_confusion_matrix(true_labels, predictions, path):
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(path)
        plt.close()
        logging.info(f"Confusion Matrix saved to {path}")

    @staticmethod
    def plot_loss_curve(train_losses, val_losses, path, epochs):
        plt.figure()
        plt.plot(range(epochs), train_losses, label='Training Loss')
        plt.plot(range(epochs), val_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.savefig(path)
        plt.close()
        logging.info(f"Loss Curve saved to {path}")

    @staticmethod
    def plot_accuracy_curve(train_accuracies, val_accuracies, path, epochs):
        plt.figure()
        plt.plot(range(epochs), [acc * 100 for acc in train_accuracies], label='Training Accuracy')
        plt.plot(range(epochs), [acc * 100 for acc in val_accuracies], label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.savefig(path)
        plt.close()
        logging.info(f"Accuracy Curve saved to {path}")

class ReportToDataFrame:
    @staticmethod
    def _classification_report_to_df(report_dict):
        report_lines = []
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                line = {
                    'Label': label,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-score': metrics['f1-score'],
                    'Support': metrics['support']
                }
                report_lines.append(line)
            else:
                line = {'Label': label, 'Precision': metrics}
                report_lines.append(line)
        report_df = pd.DataFrame(report_lines)
        return report_df

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        probas = torch.softmax(inputs, dim=1)
        pt = probas.gather(1, targets.unsqueeze(1)).squeeze(1)  # 選擇正確類別的概率 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def setup_logger(logger_name: str, log_file: str):

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # 如果還想要輸出到 console，可加一個 StreamHandler
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        # logger.addHandler(console_handler)

        # 若不想讓訊息傳到 root logger，可:
        logger.propagate = False

    return logger

class PleuralEffusionTrainer:

    def __init__(self, config_manager: ConfigManager):
        self.cfg = config_manager

        self.RESULTS_PATH = os.path.join(
            self.cfg.FILE_PATH,
            'results',
            f'{self.cfg.image_type}+{self.cfg.feature_type}_{self.cfg.model_name}',
            f'{self.cfg.timestamp}_{self.cfg.loss_function}_{self.cfg.learning_rate}_aug_{self.cfg.aug}'
        )
        os.makedirs(self.RESULTS_PATH, exist_ok=True)

        self.OUTPUT_CSV = os.path.join(
            self.cfg.FILE_PATH,
            'results',
            f'{self.cfg.image_type}+{self.cfg.feature_type}_{self.cfg.model_name}',
            '5-fold_training.csv'
        )
        self.LOG_PATH = os.path.join(self.RESULTS_PATH, 'training.log')
        self.logger = setup_logger(f'PleuralEffusionTrainer_{self.cfg.timestamp}', self.LOG_PATH)

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

    def _save_checkpoint(self, state, savepath):
        torch.save(state, savepath)
        self.logger.info(f"Model saved to {savepath}")

    def run_kfold_training(self):
        self.logger.info(f"Classification training started at {self.cfg.current_datetime} using {self.cfg.model_name}.")

        self.logger.info(
            f"""Classification
            Image Type: {self.cfg.image_type}
            Feature Type: {self.cfg.feature_type}
            Model: {self.cfg.model_name}
            Learning Rate: {self.cfg.learning_rate}
            Loss Function: {self.cfg.loss_function}
            Epochs: {self.cfg.epochs}
            Batch Size: {self.cfg.batch_size}
            Number of Folds: {self.cfg.num_folds}
            Augmentation: {self.cfg.aug}
            Number of Samples: {self.cfg.num_sample}\t 
            Number of Malignant Samples: {self.cfg.num_malignant}\t 
            Number of Benign Samples: {self.cfg.num_benign}
            """
        )

        full_train_dataset, self.test_dataset, self.clinical_features_dim = prepare_data(image_folder=self.cfg.IMAGE_FOLDER,
                                                                                         csv=self.cfg.CSV_FILE, 
                                                                                         feature_type=self.cfg.feature_type,
                                                                                         drop_columns=self.cfg.drop_columns,
                                                                                         timestamp=self.cfg.timestamp,
                                                                                         test_size=0.2, 
                                                                                         random_seed=42)
        
        kf = StratifiedKFold(n_splits=self.cfg.num_folds, shuffle=True, random_state=42)

        fold = 0
        fold_num = []
        fold_f1 = []
        fold_accuracy = []
        fold_precision = []
        fold_recall = []

        for train_indices, val_indices in kf.split(full_train_dataset, full_train_dataset.labels):
            fold += 1
            print(f"Fold {fold}/{self.cfg.num_folds}")
            self.logger.info(f"{'-'*30}Fold {fold}/{self.cfg.num_folds}{'-'*30}")

            train_image_paths = [full_train_dataset.image_paths[x] for x in train_indices]
            train_labels = [full_train_dataset.labels[x] for x in train_indices]
            train_clinical_features = [full_train_dataset.clinical_features[x] for x in train_indices]
            val_image_paths = [full_train_dataset.image_paths[x] for x in val_indices]
            val_labels = [full_train_dataset.labels[x] for x in val_indices]
            val_clinical_features = [full_train_dataset.clinical_features[x] for x in val_indices]

            augmented_images, augmented_labels, augmented_clinical_features = [], [], []
            if self.cfg.aug:
                augmented_images, augmented_labels, augmented_clinical_features = augment_data(train_image_paths, train_labels, train_clinical_features)
                self.logger.info(
                    f"Original amount of training dataset: {len(train_image_paths)}\t"
                    f"Amount of validation dataset: {len(val_image_paths)}\t"
                    f"Augmented images: {len(augmented_images)}"
                )

            mean, std = self._compute_mean_std(train_image_paths, train_labels, train_clinical_features)
            self.logger.info(f"Normalization with Mean: {mean}, Std: {std}")

            transform_normalize = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[mean[0], mean[1], mean[2]],
                    std=[std[0], std[1], std[2]]
                )
            ])

            if self.cfg.aug:
                train_dataset = PleuralEffusionDataset(
                    train_image_paths,
                    train_labels,
                    train_clinical_features,
                    transform=transform_normalize,
                    augmented_images=augmented_images,
                    augmented_labels=augmented_labels, 
                    augmented_clinical_features=augmented_clinical_features
                )
            else:
                train_dataset = PleuralEffusionDataset(
                    train_image_paths,
                    train_labels,
                    train_clinical_features, 
                    transform=transform_normalize
                )
            val_dataset = PleuralEffusionDataset(
                val_image_paths,
                val_labels,
                val_clinical_features, 
                transform=transform_normalize
            )

            train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = PleuralEffusionClassifier(self.cfg.model_name, num_classes=2, clinical_features_dim=self.clinical_features_dim)
            model.to(device)

            criterion = self._get_loss_function()
            optimizer = optim.Adam(model.parameters(), lr=self.cfg.learning_rate)
            optimizer_name = optimizer.__class__.__name__
            self.logger.info(f"Optimizer: {optimizer_name}")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            scheduler_name = scheduler.__class__.__name__
            self.logger.info(f"Scheduler: {scheduler_name}\t Step Size: {scheduler.step_size}\t Gamma: {scheduler.gamma}")

            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []

            for epoch in range(self.cfg.epochs):
                train_loss, train_accuracy = self._train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )

                val_loss, val_accuracy, filenames_list, true_labels, predictions = self._validate(
                    model, val_loader, criterion, device
                )

                accuracy = accuracy_score(true_labels, predictions)
                print(
                    f"Epoch [{epoch+1}/{self.cfg.epochs}], "
                    f"Loss: {train_loss:.4f}, Validation Accuracy: {accuracy*100:.2f}%"
                )
                self.logger.info(
                    f"Epoch [{epoch+1}/{self.cfg.epochs}], "
                    f"Loss: {train_loss:.4f}, Validation Accuracy: {accuracy*100:.2f}%"
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)

            val_df = pd.DataFrame({
                'Filename': filenames_list,
                'True Label': true_labels,
                'Predicted Label': predictions,
            })
            self.logger.info(f"Results of fold {fold} validation: \n{val_df}")

            report_dict = classification_report(
                true_labels, predictions,
                zero_division='warn',
                output_dict=True
            )
            report_df = ReportToDataFrame._classification_report_to_df(report_dict)
            self.logger.info(f"Classification Report: \n{report_df}")

            fold_num.append(fold)
            fold_f1.append(report_dict['weighted avg']['f1-score'])
            fold_accuracy.append(accuracy)
            fold_precision.append(report_dict['weighted avg']['precision'])
            fold_recall.append(report_dict['weighted avg']['recall'])

            self._save_checkpoint(
                {
                    "clinical_features_dim": self.clinical_features_dim,
                    "fold": fold,
                    "state_dict": model.state_dict(),
                    "normalization": transform_normalize, 
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "f1-score": report_dict['weighted avg']['f1-score'],
                }, f'{self.RESULTS_PATH}/checkpoint_fold_{fold}.pth'
            )

            conf_matrix_path = os.path.join(self.RESULTS_PATH, f'confusion_matrix_{fold}.png')
            loss_curve_path = os.path.join(self.RESULTS_PATH, f'loss_curve_{fold}.png')
            acc_curve_path = os.path.join(self.RESULTS_PATH, f'accuracy_curve_{fold}.png')

            Plotter.plot_confusion_matrix(true_labels, predictions, conf_matrix_path)
            Plotter.plot_loss_curve(train_losses, val_losses, loss_curve_path, self.cfg.epochs)
            Plotter.plot_accuracy_curve(train_accuracies, val_accuracies, acc_curve_path, self.cfg.epochs)

     
        time_df = pd.DataFrame({"time": [self.cfg.timestamp]})
        time_df.index = ['time']
        time_df.to_csv(self.OUTPUT_CSV, mode='a', header=True, index=False)

        final_results_df = pd.DataFrame({
            'Fold': fold_num,
            'F1 Score': fold_f1,
            'Accuracy': fold_accuracy,
            'Precision': fold_precision,
            'Recall': fold_recall,
        })
        final_results_df.to_csv(self.OUTPUT_CSV, mode='a', header=True, index=False)
        self.logger.info(f"{'='*30}Results of all folds{'='*30} \n{final_results_df}")

        self.logger.info(
            f"""Results of mean and std: 
            mean f1: {np.mean(fold_f1):.4f}\t std f1: {np.std(fold_f1):.4f}
            mean accuracy: {np.mean(fold_accuracy):.4f}\t std accuracy: {np.std(fold_accuracy):.4f}
            mean precision: {np.mean(fold_precision):.4f}\t std precision: {np.std(fold_precision):.4f}
            mean recall: {np.mean(fold_recall):.4f}\t std recall: {np.std(fold_recall):.4f}
            """
        )
        mean_std_fold_df = pd.DataFrame({
            "mean f1": [np.mean(fold_f1)],
            "std_f1": [np.std(fold_f1)],
            "mean_accuracy": [np.mean(fold_accuracy)],
            "std_accuracy": [np.std(fold_accuracy)],
            "mean_precision": [np.mean(fold_precision)],
            "std_precision": [np.std(fold_precision)],
            "mean_recall": [np.mean(fold_recall)],
            "std_recall": [np.std(fold_recall)],
        })
        mean_std_fold_df.index = ['mean & std']
        mean_std_fold_df.to_csv(self.OUTPUT_CSV, mode='a', header=True, index=False)

        return final_results_df

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

    def _train_one_epoch(self, model, train_loader, criterion, optimizer, device):
    
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, clinical_features, labels, _ in train_loader:
            images, clinical_features, labels = images.to(device), clinical_features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, clinical_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        return train_loss, train_accuracy

    def _validate(self, model, val_loader, criterion, device):
    
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        predictions = []
        true_labels = []
        filenames_list = []

        with torch.no_grad():
            for images, clinical_features, labels, filenames in val_loader:
                images, clinical_features, labels = images.to(device), clinical_features.to(device), labels.to(device)
                outputs = model(images, clinical_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                filenames_list.extend(filenames)
                true_labels.extend(labels.cpu().numpy())
                predictions.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val

        return val_loss, val_accuracy, filenames_list, true_labels, predictions

class PleuralEffusionTester:
    def __init__(self, config_manager: ConfigManager, results_path: str, test_dataset: PleuralEffusionDataset):
        self.cfg = config_manager
        self.CHECKPOINT_PATH = results_path
        self.test_dataset = test_dataset
        self.RESULTS_PATH = os.path.join(results_path, 'test_inference')
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        self.LOG_PATH = os.path.join(self.RESULTS_PATH, 'inference.log')
        self.logger = setup_logger(f'PleuralEffusionTester_{self.cfg.timestamp}', self.LOG_PATH)
        self.OUTPUT_CSV = os.path.join(
            self.cfg.FILE_PATH,
            'results',
            f'{self.cfg.image_type}+{self.cfg.feature_type}_{self.cfg.model_name}',
            '5-fold_testing_inference.csv'
        )

    def _load_checkpoint(self, fold_num: int):
        checkpoint_path = os.path.join(self.CHECKPOINT_PATH, f'checkpoint_fold_{fold_num}.pth')
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
        checkpoint = torch.load(checkpoint_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PleuralEffusionClassifier(self.cfg.model_name, num_classes=2, clinical_features_dim=checkpoint['clinical_features_dim'])
        model.to(device)
        model.load_state_dict(checkpoint['state_dict'])
        transform_normalize = checkpoint.get("normalization")
        self.logger.info(f"Normalization with {transform_normalize}")

        return model, transform_normalize, device
    
    def run_inference(self):
        self.logger.info(f"Classification testing started using {self.cfg.model_name}.")
        print(f"Classification testing started using {self.cfg.model_name}.")

        fold_nums = []
        fold_f1 = []
        fold_accuracy = []
        fold_precision = []
        fold_recall = []

        for fold_num in range(self.cfg.num_folds):
            model, transform_normalize, device = self._load_checkpoint(fold_num+1)

            test_dataset = PleuralEffusionDataset(
                image_paths = self.test_dataset.image_paths, 
                labels = self.test_dataset.labels, 
                clinical_features = self.test_dataset.clinical_features,
                transform = transform_normalize
            )
            test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size, shuffle=False)

            test_report = self._one_fold_inference(fold_num+1, model, test_loader, device)

            fold_nums.append(fold_num+1)
            fold_f1.append(test_report['weighted avg']['f1-score'])
            fold_accuracy.append(test_report['accuracy']) 
            fold_precision.append(test_report['weighted avg']['precision'])
            fold_recall.append(test_report['weighted avg']['recall'])

        time_df = pd.DataFrame({"time": [self.cfg.timestamp]})
        time_df.index = ['time']
        time_df.to_csv(self.OUTPUT_CSV, mode='a', header=True, index=False)

        final_test_results_df = pd.DataFrame({
            'Fold': fold_nums,
            'F1-Score': fold_f1,
            'Accuracy': fold_accuracy,
            'Precision': fold_precision,
            'Recall': fold_recall,
        })
        final_test_results_df.to_csv(self.OUTPUT_CSV, mode='a', header=True, index=False)
        self.logger.info(f"{'='*30}Results of all folds{'='*30} \n{final_test_results_df}")

        self.logger.info(
            f"""Results of mean and std: 
            mean f1: {np.mean(fold_f1):.4f}\t std f1: {np.std(fold_f1):.4f}
            mean accuracy: {np.mean(fold_accuracy):.4f}\t std accuracy: {np.std(fold_accuracy):.4f}
            mean precision: {np.mean(fold_precision):.4f}\t std precision: {np.std(fold_precision):.4f}
            mean recall: {np.mean(fold_recall):.4f}\t std recall: {np.std(fold_recall):.4f}
            """
        )
        test_mean_std_fold_df = pd.DataFrame({
            "mean_f1": [np.mean(fold_f1)],
            "std_f1": [np.std(fold_f1)],
            "mean_accuracy": [np.mean(fold_accuracy)],
            "std_accuracy": [np.std(fold_accuracy)],
            "mean_precision": [np.mean(fold_precision)],
            "std_precision": [np.std(fold_precision)],
            "mean_recall": [np.mean(fold_recall)],
            "std_recall": [np.std(fold_recall)],
        })
        test_mean_std_fold_df.index = ['mean & std']
        test_mean_std_fold_df.to_csv(self.OUTPUT_CSV, mode='a', header=True, index=False)
        print("Testing Done.")

        return final_test_results_df, test_mean_std_fold_df

    def _one_fold_inference(self, fold, model, test_loader, device):
        self.logger.info(f"Fold {fold} inference started.")
        model.eval()
        test_preds = []
        test_labels = []
        test_filenames = []

        with torch.no_grad():
            for images, clinical_features, labels, filenames in test_loader:
                images, clinical_features, labels = images.to(device), clinical_features.to(device), labels.to(device)
                outputs = model(images, clinical_features)
                _, predicted = torch.max(outputs.data, 1)
                
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_filenames.extend(filenames)
        
        test_report = classification_report(test_labels, test_preds, output_dict=True, zero_division='warn')
        test_report_df = ReportToDataFrame._classification_report_to_df(test_report)
        self.logger.info(f"Classification Report: \n{test_report_df}")

        conf_matrix_path = os.path.join(self.RESULTS_PATH, f'confusion_matrix_{fold}.png')
        Plotter.plot_confusion_matrix(test_labels, test_preds, conf_matrix_path)

        return test_report
             
class PleuralEffusionFinalPredictor:
    def __init__(self, config_manager: ConfigManager, results_path: str, test_results_df: pd.DataFrame):
        self.cfg = config_manager
        self.test_results_df = test_results_df
        self.CHECKPOINT_PATH = results_path
        self.RESULTS_PATH = os.path.join(results_path, 'predict_all')
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        self.LOG_PATH = os.path.join(self.RESULTS_PATH, 'predict_all.log')
        self.logger = setup_logger(f"PleuralEffusionFinalPredictor_{self.cfg.timestamp}", self.LOG_PATH)
        self.OUTPUT_CSV = os.path.join(
            self.cfg.FILE_PATH,
            'results',
            f'{self.cfg.image_type}+{self.cfg.feature_type}_{self.cfg.model_name}',
            '5-fold_predict_all.csv'
        )
        self.PRED_CSV = os.path.join(self.RESULTS_PATH, 'final_prediction.csv')

    def _select_best_fold(self) -> int:
        best_fold_idx = self.test_results_df['F1-Score'].idxmax()
        best_fold_num = self.test_results_df.loc[best_fold_idx, 'Fold']
        best_f1 = self.test_results_df.loc[best_fold_idx, 'F1-Score']
        self.logger.info(f"Best Fold: {best_fold_num}, F1-Score: {best_f1}")
        return int(best_fold_num)


    def _load_checkpoint(self, fold_num: int):
        checkpoint_path = os.path.join(self.CHECKPOINT_PATH, f'checkpoint_fold_{fold_num}.pth')
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
        checkpoint = torch.load(checkpoint_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PleuralEffusionClassifier(self.cfg.model_name, num_classes=2, clinical_features_dim=checkpoint['clinical_features_dim'])
        model.to(device)
        model.load_state_dict(checkpoint['state_dict'])
        transform_normalize = checkpoint.get("normalization")

        return model, transform_normalize, device
    
    def _build_final_dataset(self, transform):
        df = pd.read_csv(self.cfg.CSV_FILE)
        image_paths = [os.path.join(self.cfg.IMAGE_FOLDER, x) for x in os.listdir(self.cfg.IMAGE_FOLDER) if x.endswith(".jpg")]
        labels = df['Malignant'].values

        data_preprocessor = ClinicalDataPreprocessor(self.cfg.CSV_FILE, self.cfg.feature_type, self.cfg.drop_columns, self.cfg.timestamp)
        data = data_preprocessor.preprocess_data()

        final_dataset = PleuralEffusionDataset(
            image_paths = image_paths,
            labels = labels,
            clinical_features = data.drop(columns=['Malignant']).values,
            transform = transform
        )

        return final_dataset

    def _predict_all(self):
        self.logger.info("Final Prediction started.")        
        best_fold_num = self._select_best_fold()
        model, transform_normalize, device = self._load_checkpoint(best_fold_num)
        self.logger.info(f"Using transformed normalization: {transform_normalize}")

        final_dataset = self._build_final_dataset(transform_normalize)
        final_loader = DataLoader(final_dataset, batch_size=self.cfg.batch_size, shuffle=False)

        all_preds = []
        all_labels = []
        all_filenames = []

        model.eval()
        with torch.no_grad():
            for images, clinical_features, labels, filenames in final_loader:
                images, clinical_features, labels = images.to(device), clinical_features.to(device), labels.to(device)
                outputs = model(images, clinical_features)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_filenames.extend(filenames)
        
        all_preds_df = pd.DataFrame({
            'Filename': all_filenames,
            'Predicted Label': all_preds,
            'True Label': all_labels
        })
        self.logger.info(f"Final Prediction Results Saved to {self.PRED_CSV}")
        all_preds_df.to_csv(self.PRED_CSV, index=False)

        final_report = classification_report(all_labels, all_preds, output_dict=True, zero_division='warn')
        final_report_df = ReportToDataFrame._classification_report_to_df(final_report)
        self.logger.info(f"Classification Report: \n{final_report_df}")

        Plotter.plot_confusion_matrix(all_labels, all_preds, os.path.join(self.RESULTS_PATH, f'confusion_matrix_fold_{best_fold_num}.png'))
        
        time_df = pd.DataFrame({
            "time": [self.cfg.timestamp], 
            "fold": [best_fold_num]
        })
        time_df.index = ['time']
        time_df.to_csv(self.OUTPUT_CSV, mode='a', header=True, index=False)

        final_report_df.to_csv(self.OUTPUT_CSV, mode='a', header=True, index=False)
        print("Predict done.")

def run_all_combinations(config_path='Classification/DL/image_clinical/config.yaml'):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    combo_logger = setup_logger(
        logger_name="AllParameterCombinations", 
        log_file=f"Classification/DL/image_clinical/results/{timestamp}_all_parameter_combinations.log"
    )
    combo_logger.info(f"{'-'*30} {timestamp} Run all combinations started. {'-'*30}")

    with open(config_path, 'r') as file:
        base_config = yaml.safe_load(file)

    model_options = base_config['model_parameter']['model_name']['options']
    lr_options = base_config['model_parameter']['learning_rate']['options']
    loss_options = base_config['model_parameter']['loss_function']['options']
    image_type_options = base_config['data_setting']['image_type']['options']
    feature_type_options = base_config['data_setting']['feature_type']['options']
    aug_options = base_config['data_setting']['aug']['options']

    param_combinations = itertools.product(
        image_type_options, feature_type_options, model_options, loss_options, lr_options, aug_options
    )

    for img_type, feature_type, model_name, loss_func, lr, aug in param_combinations:
        print(f"{'='*60}\nRunning parameters: image_type = {img_type}, feature_type = {feature_type}, model = {model_name}, loss function = {loss_func}, lr = {lr}, aug = {aug}")
        config_copy = dict(base_config)

        config_copy['model_parameter']['model_name']['default'] = model_name
        config_copy['model_parameter']['learning_rate']['default'] = lr
        config_copy['model_parameter']['loss_function']['default'] = loss_func
        config_copy['data_setting']['image_type']['default'] = img_type
        config_copy['data_setting']['feature_type']['default'] = feature_type
        config_copy['data_setting']['aug']['default'] = aug

        config_manager = ConfigManager(config=config_copy)

        trainer = PleuralEffusionTrainer(config_manager)
        trainer.run_kfold_training()

        tester = PleuralEffusionTester(
            config_manager=config_manager,
            results_path=trainer.RESULTS_PATH,
            test_dataset=trainer.test_dataset
        )
        test_results_df,  test_mean_std_fold_df= tester.run_inference()
        combo_logger.info(f"""
            {'='*70}
            Time: {trainer.cfg.timestamp}
            Running parameters: 
            Image_type = {img_type}
            Feature_type = {feature_type}
            Model = {model_name}
            Loss function = {loss_func}
            Learning rate = {lr}
            Augmentation = {aug}
        """)        
        combo_logger.info(f"Test Results: \n{test_results_df}\n")
        combo_logger.info(f"Test Mean & Std: \n{test_mean_std_fold_df}\n")

        predictor = PleuralEffusionFinalPredictor(
            config_manager=config_manager,
            results_path=trainer.RESULTS_PATH,
            test_results_df=test_results_df
        )
        predictor._predict_all()

    print(f"All parameters combinations completed")
    combo_logger.info(f"All parameters combinations completed\n{'*'*100}\n{'*'*100}\n{'*'*100}\n")

def main(run_all=False):
    config_path = 'Classification/DL/image_clinical/config.yaml'  # 你的 config 路徑

    if run_all:
        run_all_combinations(config_path)
    else:
        config_manager = ConfigManager(config_path)
        trainer = PleuralEffusionTrainer(config_manager)
        trainer.run_kfold_training()

        tester = PleuralEffusionTester(
            config_manager=config_manager,
            results_path=trainer.RESULTS_PATH,
            test_dataset=trainer.test_dataset
        )
        test_results_df = tester.run_inference()

        predictor = PleuralEffusionFinalPredictor(
            config_manager=config_manager,
            results_path=trainer.RESULTS_PATH,
            test_results_df=test_results_df
        )
        predictor._predict_all()


if __name__ == "__main__":
    main(run_all=True)