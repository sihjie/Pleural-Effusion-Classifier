import os
import seaborn as sns
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import shap
from matplotlib.lines import Line2D

from Classification.DL_ML.data_preparation import ClinicalDataPreprocessor

matplotlib.use("Agg")  # 非互動式，不用 tkinter，就不會有錯誤

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

    def plot_pca(labels, reduced_features, path):
        plt.figure(figsize=(8, 6))
        cmap = mcolors.ListedColormap(['blue', 'red'])

        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap=cmap, alpha=0.5)
        # 自訂圖例：定義兩個圖例項目
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Benign',
                markerfacecolor='blue', markersize=8, alpha=0.5),
            Line2D([0], [0], marker='o', color='w', label='Malignant',
                markerfacecolor='red', markersize=8, alpha=0.5)
        ]

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA of Deep Features")
        plt.legend(handles=legend_elements)
        plt.savefig(path)
        plt.close()
        logging.info(f"PCA image saved to {path}")

    def plot_kpca(labels, reduced_features, path):
        plt.figure(figsize=(8, 6))
        for label in np.unique(labels):
            idx = labels == label
            plt.scatter(
                reduced_features[idx,0], reduced_features[idx,1],
                label=f"Class {label}", alpha=0.7
            )
        plt.xlabel("KPCA Component 1")
        plt.ylabel("KPCA Component 2")
        plt.title("KPCA 2D Embedding")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logging.info(f"KPCA image saved to {path}")

    @staticmethod
    def plot_embedding(dr_method, labels, embedding, path):
        plt.figure(figsize=(8,6))
        for lab in np.unique(labels):
            idx = labels == lab
            plt.scatter(embedding[idx,0], embedding[idx,1], label=str(lab), alpha=0.7)
        plt.legend()
        plt.title("Embedding via " + dr_method)
        plt.savefig(path)
        plt.close()
        logging.info(f"Embedding image saved to {path}")

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
    
class MetricsToDatafram:
    @staticmethod
    def _model_metrics_to_df(model_metrics):
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

class SHAPPlotter:
    def __init__(self, datapreprocessor: ClinicalDataPreprocessor, x_train, predict, path, model_name, fold_index):
        self.dp = datapreprocessor
        self.original_columns = self.dp.before_ohe_columns.drop('Malignant')
        df_cols = [f'DF{i}' for i in range(1, 46)]
        self.all_columns = pd.Index(list(self.original_columns) + df_cols)  # clinical features + deep features columns
        self.x_train = x_train
        self.predict = predict
        self.label_names = ['Benign', 'Malignant']
        self.path = path
        self.model_name = model_name
        self.fold_index = fold_index

        if self.dp and hasattr(self.dp, 'after_ohe_data'):
            self.ohe_columns = self.dp.ohe_columns
            self.after_ohe_columns = self.dp.after_ohe_data.columns.drop('Malignant')
            self.after_ohe_columns_df = pd.Index(list(self.after_ohe_columns) + df_cols)  # 聚合後的特徵名稱
            self.encoder = self.dp.encoder  
        else:
            self.encoder = None  

    def plot_shape_values(self):
        if self.encoder:
            self.ohe_feature_names = self.encoder.get_feature_names_out()

        explainer = shap.KernelExplainer(self.predict, self.x_train)

        shap_values = explainer.shap_values(self.x_train)
        if self.encoder:
            shap_values_aggregated = self.aggregate_shap_values(shap_values, self.after_ohe_columns_df)

            # 聚合特徵數據（使其與聚合後的 SHAP 值一致）
            aggregated_features = pd.DataFrame(0, index=np.arange(self.x_train.shape[0]), columns=self.all_columns)  # 創建一個空的 DataFrame 來存儲聚合的特徵值

            for col in self.ohe_columns:
                encoded_feature_names = [name for name in self.ohe_feature_names if col in name]
                aggregated_features[col] = (self.x_train[:, self.after_ohe_columns_df.isin(encoded_feature_names)]*np.arange(len(encoded_feature_names))).sum(axis=1)  # 將特徵值乘以0-n（以區分類別）再相加
            
                non_ohe_columns = self.all_columns.difference(self.ohe_columns)  # 獲取非 One-Hot Encoding 的特徵名稱

                column_indices = [self.all_columns.get_loc(col) for col in non_ohe_columns]    # 獲取非 One-Hot Encoding 特徵的索引位置
                selected_features = self.x_train[:, column_indices]      # 根據索引位置選擇 X_train 中相應欄位，保持原有順序
                aggregated_features[non_ohe_columns] = selected_features        # 將對應的特徵值賦值給 aggregated_features 中相應的欄位
            
            shap.summary_plot(shap_values_aggregated.values, aggregated_features, plot_type="bar", class_names = self.label_names, feature_names=self.all_columns, show=False)
            plt.savefig(f'{self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary.png')
            plt.close()
            logging.info(f"SHAP Summary Plot saved to {self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary.png")
            shap.summary_plot(shap_values_aggregated.values, aggregated_features, feature_names=self.all_columns, show=False)    
            plt.savefig(f'{self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary_for_Malignant.png')
            logging.info(f"SHAP Summary Plot for Malignant saved to {self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary_for_Malignant.png")
            plt.close()
        else:
            shap.summary_plot(shap_values, self.x_train, plot_type="bar", class_names = self.label_names, feature_names=self.all_columns, show=False)
            plt.savefig(f'{self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary.png')
            plt.close()
            logging.info(f"SHAP Summary Plot saved to {self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary.png")
            shap.summary_plot(shap_values, self.x_train, feature_names=self.all_columns, show=False)
            plt.savefig(f'{self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary_for_Malignant.png')
            logging.info(f"SHAP Summary Plot for Malignant saved to {self.path}/{self.model_name}_fold-{self.fold_index}_shap_summary_for_Malignant.png")
            plt.close()

    def aggregate_shap_values(self, shap_values, feature_names):
        shap_values_aggregated = pd.DataFrame(0, index=np.arange(shap_values.shape[0]), columns=self.all_columns)  # 創建一個空的 DataFrame 來存儲聚合的 SHAP 值
                
        for column in self.ohe_columns:
            encoded_feature_names = [name for name in self.ohe_feature_names if column in name]  # 逐一獲取含'column'的 One-Hot Encoding 的特徵名稱
            shap_values_subset = shap_values[:, feature_names.isin(encoded_feature_names)]  # 從所有 SHAP 值中獲取相應的 SHAP 值
            shap_values_aggregated[column] = shap_values_subset.sum(axis=1) # 將 SHAP 值相加

        non_ohe_columns = self.all_columns.difference(self.ohe_columns)  # 獲取非 One-Hot Encoding 的特徵名稱
        shap_values_aggregated[non_ohe_columns] = shap_values[:, feature_names.isin(non_ohe_columns)]

        return shap_values_aggregated
