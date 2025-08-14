import os
import yaml
import logging
import pandas as pd

from Classification.DL_ML.logger_utils import setup_log

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
                
        self.model_name = self._config['DL_model']['model_name']['default']
        self.learning_rate = self._config['DL_model']['learning_rate']['default']
        self.loss_function = self._config['DL_model']['loss_function']['default']
        self.epochs = self._config['DL_model']['epochs']
        self.batch_size = self._config['DL_model']['batch_size']
        self.early_stop = self._config['DL_model']['early_stopping']['default']
        self.patience = self._config['DL_model']['early_stopping']['patience']
        self.dropout = self._config['DL_model']['dropout']['default']
        self.dropout_rate = self._config['DL_model']['dropout']['dropout_rate']

        self.image_type = self._config['data_setting']['image_type']['default']
        self.feature_type = self._config['data_setting']['feature_type']['default']
        self.drop_columns = self._config['data_setting']['drop_columns']
        self.aug = self._config['data_setting']['aug']['default']

        self.dimred_method = self._config['dimred']['default']

        self.num_sample = self._config['data_setting']['num_sample']
        self.num_malignant = self._config['data_setting']['num_malignant']
        self.num_benign = self.num_sample - self.num_malignant

        self.current_datetime = pd.Timestamp.now()
        self.timestamp = self.current_datetime.strftime("%Y%m%d_%H%M%S")
        self.FILE_PATH = 'Classification/DL_ML'
        self.IMAGE_FOLDER = self._determine_image_folder()
        self.CSV_FILE = os.path.join('Classification', f"{self._config['data_setting']['feature_type']['default']}.csv")

        self.RESULTS_FOLDER = os.path.join(self.FILE_PATH, 'results', 'ok version', 
                                           f'{self.image_type} + {self.feature_type} --{self.model_name}', 
                                           f'{self.timestamp} {self.loss_function}_{self.learning_rate} {self.dimred_method}')
        os.makedirs(self.RESULTS_FOLDER, exist_ok=True)
        self.LOG_PATH = os.path.join(self.RESULTS_FOLDER, f"training_logs.log")
        self.confusion_matrix_output_dir = self.RESULTS_FOLDER + "/confusion_matrix_fig/"
        self.ROC_output_dir = self.RESULTS_FOLDER + "/ROC_fig/"
        self.shap_output_dir = self.RESULTS_FOLDER + "/shap_fig/"
        os.makedirs(self.confusion_matrix_output_dir, exist_ok=True)
        os.makedirs(self.ROC_output_dir, exist_ok=True)
        os.makedirs(self.shap_output_dir, exist_ok=True)
        self._write_config_to_log()

    def _load_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def __getitem__(self, key):
        return self._config[key]
    
    def _determine_image_folder(self):
        if self.image_type == 'original':
            return 'Preprocessing/Cropped'
        elif self.image_type == 'CLAHE':
            return 'Preprocessing/CLAHE'
        elif self.image_type == 'bm3d_clahe':
            return 'Preprocessing/BM3D_CLAHE'
        elif self.image_type == 'expand_15':
            return 'Preprocessing/expand_15_roi'
        elif self.image_type == 'expand_20':
            return 'Preprocessing/expand_20_roi'
        elif self.image_type == 'bm3d':
            return 'Preprocessing/BM3D'
        elif self.image_type == 'x-ray':
            return r'C:\Users\ashle\Ashlee\BusLab\workkkk\RawDataset\X-Ray'
        else:
            raise ValueError(f"Unknown image_type: {self.image_type}")
    
    def _write_config_to_log(self):  
        setup_log(self.LOG_PATH)    
        logging.info(f"Training started at {self.current_datetime}")
        logging.info("Configuration settings:")
        logging.info(
            f"""Classification
            Input data setting:
            Image Type: {self.image_type}
            Feature Type: {self.feature_type}

            Deep Learning Model for Feature Extraction setting: 
            Deep Learning Model: {self.model_name}
            Augmentation: {self.aug}
            Learning Rate: {self.learning_rate}
            Loss Function: {self.loss_function}
            Epochs: {self.epochs}
            Batch Size: {self.batch_size}    
            Early Stopping: {self.early_stop}   
            Patience: {self.patience} 
            Dropout: {self.dropout}
            Dropout Rate: {self.dropout_rate}    

            Dimensionality Reduction Method: {self.dimred_method}
            
            Number of Samples: {self.num_sample}\t 
            Number of Malignant Samples: {self.num_malignant}\t 
            Number of Benign Samples: {self.num_benign}
            """
        )
