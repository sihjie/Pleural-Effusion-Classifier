import yaml
from datetime import datetime
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import shap
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

class ConfigManager:
    def __init__(self, config_path=None):
        if config_path is not None:
            self.config_path = config_path
            self._config = self._load_config()
        else:
            raise ValueError("Either config_path or config must be provided")
        
        self.input = self._config['data']['input']['default']
        self.csv_path = f'Classification/{self.input}.csv'
        self.data = pd.read_csv(self.csv_path)
        self.smote = self._config['data']['SMOTE']['default']
        self.tl = self._config['data']['TomekLinks']['default']
        self.drop_columns = self._config['data']['drop_columns']

        # self.model = self._config['model']['name']['default']

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.RESULT_FOLD = f'Classification/ML/results/{self.input}'
        self.RESULT_PATH = f'{self.RESULT_FOLD}/{self.timestamp}'
        os.makedirs(self.RESULT_PATH, exist_ok=True)

        # self.LOG_PATH = f'{self.RESULT_PATH}/logs/{self.timestamp}.log'
        self.LOG_PATH = self.RESULT_PATH + "/ML_training_logger.log"
        self.shap_output_dir = self.RESULT_PATH + "/shap_fig/"
        self.confusion_matrix_output_dir = self.RESULT_PATH + "/confusion_matrix_fig/"
        self.ROC_output_dir = self.RESULT_PATH + "/ROC_fig/"
        os.makedirs(self.shap_output_dir, exist_ok=True)
        os.makedirs(self.confusion_matrix_output_dir, exist_ok=True)
        os.makedirs(self.ROC_output_dir, exist_ok=True)

    def _load_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def __getitem__(self, key):
        return self._config[key]
    
class Plotter:
    @staticmethod
    def plot_confusion_matrix(true_labels, predictions, path, model_name, fold_index):
        save_path = os.path.join(path, f'{model_name}_fold-{fold_index}_confusion_matrix.png')
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
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Confusion Matrix saved to {save_path}")

    @staticmethod
    def plot_roc_curve(test, pred_probs, path, model_name, fold_index):
        save_path = os.path.join(path, f'{model_name}_fold-{fold_index}_roc_curve.png')
        fpr, tpr, _ = roc_curve(test, pred_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # 隨機預測的基準線
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(save_path)
        plt.close()
        logging.info(f"ROC Curve saved to {save_path}")

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
    
def setup_log(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # 確保日誌目錄存在

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)  # 清除所有舊的 handlers

    logging.basicConfig(
        filename=filename,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger('shap').setLevel(logging.WARNING)  # 禁用 SHAP 的 INFO 級別日誌
    print(f"Log file created at {filename}")
    

class ClinicalDataPreprocessor:
    def __init__(self, config_manager: ConfigManager):
        self.cfg = config_manager
        self.data = self.cfg.data
        self.ohe_columns = ["Appearance", "Color"]
        self.encoder = OneHotEncoder(sparse_output=False)

    def preprocess_data(self):
        logging.info("----> Data Preprocessing started.\n")
        logging.info(f"Original data columns:\n{self.data.columns}\n")
        if 'clinical' in self.cfg.input:
            self._drop_columns()
            logging.info(f"After dropping columns: \n{self.data.columns}\n")
            self.before_ohe_columns = self.data.columns

            self._data_mapping()

            self.after_ohe_data = self._one_hot_encoding()
            logging.info(f"After One-Hot Encoding: \n{self.after_ohe_data.columns}\n")

            self._fill_missing_values()
        else:
            self.data = self.data.drop(columns=['filename'])
            logging.info("""
                         drop 'filename' column
                         No data preprocessing required for this dataset because not contained clinical features.
                         """)
            self.before_ohe_columns = self.data.columns
        logging.info(f"---- Data Preprocessing completed. ----\n\n")
        return self.data

    def _drop_columns(self):
        drop_columns = self.cfg.drop_columns
        self.data = self.data.drop(columns=drop_columns, errors='ignore')
        logging.info(f"Columns dropped: {drop_columns}")

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

class DatasetHandler:
    def __init__(self, config_manager: ConfigManager, data):
        self.cfg = config_manager
        self.data = data
        self.smote = self.cfg.smote
        self.tl = self.cfg.tl
    
    def prepare_data(self):
        logging.info("----> Data Preparing started.")
        x_train, x_test, y_train, y_test = self._split_data()
        x_train, y_train = self._balance_data(x_train, y_train)
        x_train, x_test = self._scale_feature(x_train, x_test)
        return x_train, x_test, y_train, y_test

    def _split_data(self):
        x = self.data.drop(columns=['Malignant']).values
        y = self.data['Malignant'].values
        logging.info(f"Data split by 80:20")
        return train_test_split(x, y, test_size=0.2, random_state=42)
    
    def _balance_data(self, x_train, y_train):
        logging.info(f"Data balanced by \tSMOTE: {self.smote}\t TomekLinks: {self.tl}")
        logging.info(f"Before SMOTE & TomekLinks: {x_train.shape}")
        if self.smote:
            smote = SMOTE(random_state=42)
            x_train, y_train = smote.fit_resample(x_train, y_train)
        logging.info(f"After SMOTE: {x_train.shape}")
        if self.tl:
            tl = TomekLinks()
            x_train, y_train = tl.fit_resample(x_train, y_train)
        logging.info(f"After TomekLinks: {x_train.shape}")
        return x_train, y_train
    
    def _scale_feature(self, x_train, x_test):
        scaler = StandardScaler()
        logging.info(f"Training feature scaled using StandardScaler\n\n")
        return scaler.fit_transform(x_train), scaler.transform(x_test)

class ModelTrainer:
    def __init__(self, config_manager: ConfigManager, data_preprocessor: ClinicalDataPreprocessor, data):
        self.cfg = config_manager
        self.dp = data_preprocessor
        self.data = data
        self.smote = self.cfg.smote
        self.tl = self.cfg.tl

    def run_nested_cv(self): 
        X = self.data.drop(columns=['Malignant']).values
        y = self.data['Malignant'].values

        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model_names = self.cfg['model']['name']['options']
        model_metrics = {
            model_name: {
                'f1': [],
                'accuracy': [],
                'precision': [],
                'recall': []
            }
            for model_name in model_names
        }

        fold_index = 1
        for train_idx, test_idx in outer_cv.split(X, y):
            logging.info('\n\n')
            logging.info(f"========== [Outer Fold {fold_index}] ==========\n")
            
            # 2. 切割該 fold 的 train/test
            x_train, x_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 3. 資料平衡 (SMOTE, TomekLinks) 與特徵縮放
            x_train, y_train = self._balance_data(x_train, y_train)
            x_train, x_test = self._scale_feature(x_train, x_test)

            # 4. 訓練 + 測試 (對所有指定的 model)
            logging.info("----> Training all models started.\n")
            for model_name in self.cfg['model']['name']['options']:
                best_model, model = self._train_one_model(x_train, y_train, model_name)

                logging.info(f"--- Evaluating {model_name} model started. ---")
                f1, acc, precision, recall = self._evaluate_model(
                    x_train, x_test, y_test, best_model, model_name, fold_index
                )

                model_metrics[model_name]['f1'].append(f1)
                model_metrics[model_name]['accuracy'].append(acc)
                model_metrics[model_name]['precision'].append(precision)
                model_metrics[model_name]['recall'].append(recall)

            logging.info(f"Fold {fold_index} all models trained and evaluated complete.")

            fold_index += 1

        # 5) 全部外層 fold 結束後，針對每個模型計算 mean & std
        logging.info("===== Cross Validation Results (Test) =====")
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
            logging.info(f'{"-"*50}\n')


    # def train_all_models(self):
    #     logging.info("----> Training all models started.\n")
    #     for model_name in self.cfg['model']['name']['options']:
    #         best_model, model = self._train_one_model(model_name)

    #         logging.info(f"--- Evaluating {model_name} model started. ---")
    #         self._evaluate_model(best_model, model, model_name)
    #     logging.info("All models trained and evaluated complete.")

    def _get_model_and_params(self, model_name=None):
        logging.info(f"{'#'*30}")
        logging.info(f"Using model: {model_name}")
        model_params = self.cfg['model']['parameters'][model_name]
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
    
    def _train_one_model(self, x_train, y_train, model_name):
        self.now_model, self.param_grid = self._get_model_and_params(model_name)
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
        cv_results[columns_to_log].to_csv(f'{self.cfg.RESULT_PATH}/{model_name}_cv_results.csv')

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        best_index = grid_search.best_index_
        best_metrics = cv_results.loc[best_index, columns_to_log]
        logging.info(f"Best Parameters: {best_params}\nBest metrics: \n{best_metrics}")
        logging.info(f"--- {model_name} model trained successfully. ---\n")
        
        return best_model, self.now_model
    
    def _evaluate_model(self, x_train, x_test, y_test, best_model, model_name, fold_index):
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
    
    def _balance_data(self, x_train, y_train):
        logging.info(f"Data balanced by \tSMOTE: {self.smote}\t TomekLinks: {self.tl}")
        logging.info(f"Before SMOTE & TomekLinks: {x_train.shape}")
        if self.smote:
            smote = SMOTE(random_state=42)
            x_train, y_train = smote.fit_resample(x_train, y_train)
        logging.info(f"After SMOTE: {x_train.shape}")
        if self.tl:
            tl = TomekLinks()
            x_train, y_train = tl.fit_resample(x_train, y_train)
        logging.info(f"After TomekLinks: {x_train.shape}")
        return x_train, y_train
    
    def _scale_feature(self, x_train, x_test):
        scaler = StandardScaler()
        logging.info(f"Training feature scaled using StandardScaler\n\n")
        return scaler.fit_transform(x_train), scaler.transform(x_test)

class SHAPPlotter:
    def __init__(self, datapreprocessor: ClinicalDataPreprocessor, x_train, predict, path, model_name, fold_index):
        self.dp = datapreprocessor
        self.original_columns = self.dp.before_ohe_columns.drop('Malignant')
        self.x_train = x_train
        self.predict = predict
        self.label_names = ['Benign', 'Malignant']
        self.path = path
        self.model_name = model_name
        self.fold_index = fold_index

        if self.dp and hasattr(self.dp, 'after_ohe_data'):
            self.ohe_columns = self.dp.ohe_columns
            self.after_ohe_columns = self.dp.after_ohe_data.columns.drop('Malignant')
            self.encoder = self.dp.encoder  
        else:
            self.encoder = None  

    def plot_shape_values(self):
        if self.encoder:
            self.ohe_feature_names = self.encoder.get_feature_names_out()

        explainer = shap.KernelExplainer(self.predict, self.x_train)

        shap_values = explainer.shap_values(self.x_train)
        if self.encoder:
            shap_values_aggregated = self.aggregate_shap_values(shap_values, self.after_ohe_columns)

            # 聚合特徵數據（使其與聚合後的 SHAP 值一致）
            aggregated_features = pd.DataFrame(0, index=np.arange(self.x_train.shape[0]), columns=self.original_columns)  # 創建一個空的 DataFrame 來存儲聚合的特徵值

            for col in self.ohe_columns:
                encoded_feature_names = [name for name in self.ohe_feature_names if col in name]
                aggregated_features[col] = (self.x_train[:, self.after_ohe_columns.isin(encoded_feature_names)]*np.arange(len(encoded_feature_names))).sum(axis=1)  # 將特徵值乘以0-n（以區分類別）再相加
            
                non_ohe_columns = self.original_columns.difference(self.ohe_columns)  # 獲取非 One-Hot Encoding 的特徵名稱

                column_indices = [self.after_ohe_columns.get_loc(col) for col in non_ohe_columns]    # 獲取非 One-Hot Encoding 特徵的索引位置
                selected_features = self.x_train[:, column_indices]      # 根據索引位置選擇 X_train 中相應欄位，保持原有順序
                aggregated_features[non_ohe_columns] = selected_features        # 將對應的特徵值賦值給 aggregated_features 中相應的欄位
        
                shap.summary_plot(shap_values_aggregated.values, aggregated_features, plot_type="bar", class_names = self.label_names, feature_names=self.original_columns, show=False)
                plt.savefig(f'{self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary.png')
                plt.close()
                logging.info(f"SHAP Summary Plot saved to {self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary.png")
                shap.summary_plot(shap_values_aggregated.values, aggregated_features, feature_names=self.original_columns, show=False)    
                plt.savefig(f'{self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary_for_Malignant.png')
                logging.info(f"SHAP Summary Plot for Malignant saved to {self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary_for_Malignant.png")
                plt.close()
        else:
            shap.summary_plot(shap_values, self.x_train, plot_type="bar", class_names = self.label_names, feature_names=self.original_columns, show=False)
            plt.savefig(f'{self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary.png')
            plt.close()
            logging.info(f"SHAP Summary Plot saved to {self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary.png")
            shap.summary_plot(shap_values, self.x_train, feature_names=self.original_columns, show=False)
            plt.savefig(f'{self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary_for_Malignant.png')
            logging.info(f"SHAP Summary Plot for Malignant saved to {self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary_for_Malignant.png")
            plt.close()

    def aggregate_shap_values(self, shap_values, feature_names):
        shap_values_aggregated = pd.DataFrame(0, index=np.arange(shap_values.shape[0]), columns=self.original_columns)  # 創建一個空的 DataFrame 來存儲聚合的 SHAP 值
        
        for column in self.ohe_columns:
            encoded_feature_names = [name for name in self.ohe_feature_names if column in name]  # 逐一獲取含'column'的 One-Hot Encoding 的特徵名稱
            shap_values_subset = shap_values[:, feature_names.isin(encoded_feature_names)]  # 從所有 SHAP 值中獲取相應的 SHAP 值
            shap_values_aggregated[column] = shap_values_subset.sum(axis=1) # 將 SHAP 值相加

        non_ohe_columns = self.original_columns.difference(self.ohe_columns)  # 獲取非 One-Hot Encoding 的特徵名稱
        shap_values_aggregated[non_ohe_columns] = shap_values[:, feature_names.isin(non_ohe_columns)]

        return shap_values_aggregated
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # config_path = "Classification/ML/config.yaml"
    config_manager = ConfigManager(args.config)
    setup_log(config_manager.LOG_PATH)
    logging.info(f"Classification using Machine Learning with input csv file: {config_manager.input}.csv started.")

    data_preprocessor = ClinicalDataPreprocessor(config_manager)
    data = data_preprocessor.preprocess_data()
    dataset_handler = DatasetHandler(config_manager, data)
    # x_train, x_test, y_train, y_test = dataset_handler.prepare_data()
    
    model_trainer = ModelTrainer(config_manager, data_preprocessor, data)
    model_trainer.run_nested_cv()

if __name__ == '__main__':  
    main()
    logging.shutdown()