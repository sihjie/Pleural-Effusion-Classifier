import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold
import yaml
import argparse

import segmentation_models_pytorch as smp
from nnunet.network_architecture.generic_UNet import Generic_UNet
from monai.networks.nets import AttentionUnet

from Segmentation.GLFR_main.src.GLFRNet import GLFRNet
from Segmentation.data_loader import prepare_data, augment_data, SegmentationDataset


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
        
        ms = self._config['model_setting']
        ds = self._config['data_setting']

        # model settings
        self.model_name = ms['model_name']['default']
        self.num_folds = ms['num_folds']
        self.patience = ms['early_stopping']['patience']
        self.epochs = ms['epochs']['default']
        self.batch_size = ms['batch_size']['default']
        self.learning_rate = ms['learning_rate']['default']
        self.loss_function = ms['loss_function']['default']

        # data settings
        self.image_type = ds['img_type']['default']
        self.aug = ds['aug']['default']

        # runtime info
        self.current_datetime = pd.Timestamp.now()
        self.timestamp = self.current_datetime.strftime("%Y%m%d_%H%M%S")

        # directories
        self.output_dir = f'Segmentation/results/{self.model_name}___{self.image_type}_{self.loss_function}/{self.timestamp}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.MASK_FOLDER = 'Preprocessing/Mask'
        self.CSV_FILE = 'Classification/clinical_feature.csv'
        self.log_path = f'{self.output_dir}/training.log'
        self.output_csv = f'Segmentation/results/{self.model_name}___{self.image_type}/cross_validation_results.csv'
        print(f'--- Running {self.model_name} with {self.image_type} images ---')

    def _load_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def write_config_to_log(self):  
        logging.info(f"Training started at {self.current_datetime}")
        logging.info("Configuration settings:")
        logging.info(
            f"""model setting:
                Model Name: {self.model_name}
                Number of Folds: {self.num_folds}
                Early Stopping: {self.patience}
                Epochs: {self.epochs}
                Batch Size: {self.batch_size}
                Learning Rate: {self.learning_rate}
                Loss Function: {self.loss_function}
            """)
        logging.info(
            f"""data setting:
                Image Type: {self.image_type}
                Augmentation: {self.aug}
                """)

    def get_image_folder(self):
        mapping = {
            'bm3d_clahe': 'Preprocessing/BM3D_CLAHE',
            'original': 'Preprocessing/Cropped',
            'bm3d': 'Preprocessing/BM3D', 
            'clahe': 'Preprocessing/CLAHE',
        }
        return mapping[self.image_type]
    
    def build_model(self):
        if self.model_name == 'UNet++':
            return smp.UnetPlusPlus(
                        encoder_name="resnet34",        # 使用的backbone，可以根據需要選擇不同的編碼器
                        encoder_weights="imagenet",     # 預訓練的權重
                        in_channels=3,                  # 輸入通道數，通常RGB圖像是3
                        classes=2 
                    )
        
        elif self.model_name == 'GLFR':
            return GLFRNet(n_class=2, backbone='resnet34', aux=True, pretrained_base=True)
        
        elif self.model_name == 'UNet':
            return smp.Unet(
                        encoder_name="resnet34",        # 使用的backbone，可以根據需要選擇不同的編碼器
                        encoder_weights="imagenet",     # 預訓練的權重
                        in_channels=3,                  # 輸入通道數，通常RGB圖像是3
                        classes=2 
                    )
        elif self.model_name == 'AttentionUNet':
            return AttentionUnet(
                spatial_dims=2,               # 二維分割
                in_channels=3,                # RGB 三通道輸入
                out_channels=2,               # 輸出類別數
                channels=(64, 128, 256, 512), # 每層 feature map 通道數
                strides=(2, 2, 2, 2),         # 對應的下採樣大小
                kernel_size=3,                # conv kernel size
                up_kernel_size=3,             # 轉置 conv kernel size
                dropout=0.0                   # 若需 dropout，可調整此值
            )  # :contentReference[oaicite:0]{index=0}
        
        elif self.model_name == 'nnUNet':
            return Generic_UNet(
                input_channels=3,       # 三通道輸入
                base_num_features=64,   # 第一層特徵圖數量
                num_classes=2,          # 分割類別數
                num_pool=4,             # 下採樣階數
                num_conv_per_stage=2,   # 每個階段的卷積層數
                feat_map_mul_on_downscale=2,  # 每下採樣一次，特徵圖數量乘上的倍數
                conv_op=nn.Conv2d,
                norm_op=nn.InstanceNorm2d,
                dropout_op=nn.Dropout2d,
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={'negative_slope':1e-2,'inplace':True},
                deep_supervision=False,  # 是否使用深度監督
            ) 
        
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

class CheckpointManager:
    def __init__(self, base_dir):
        self.checkpoint_dir = base_dir

    def save(self, state, fold):
        path = os.path.join(self.checkpoint_dir, f"ckpt_fold_{fold}.pth")
        torch.save(state, path)
        logging.info(f"Saved best model for fold {fold} at {path}")
        return path
                
class DataManager:
    def __init__(self, cfg: ConfigManager):  
        self.image_dir = cfg.get_image_folder()
        self.mask_dir = cfg.MASK_FOLDER
        self.batch_size = cfg.batch_size
        self.use_aug = cfg.aug   

    def prepare_fold(self, train_idx, val_idx):
        full_ds, _ = prepare_data(self.image_dir, self.mask_dir)
        train_paths = [full_ds.image_paths[i] for i in train_idx]
        train_masks = [full_ds.mask_paths[i] for i in train_idx]
        val_paths = [full_ds.image_paths[i] for i in val_idx]
        val_masks = [full_ds.mask_paths[i] for i in val_idx]

        # augment train
        aug_imgs, aug_masks = [], []
        if self.use_aug:
            aug_imgs, aug_masks = augment_data(train_paths, train_masks)

        logging.info(
            f"""Original training images amount:{len(train_paths)}
                Augmented training images amount:{len(aug_imgs)}
                Total training images amount:{len(train_paths) + len(aug_imgs)}
            """)

        # compute normalization stats
        logging.info(f"Computing normalization stats for training images (only original training images)")
        mean, std, self.weight = self._compute_stats(train_paths, train_masks)
        self.norm_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_ds = SegmentationDataset(
            train_paths, train_masks,
            transform=self.norm_transform,
            augmented_images=aug_imgs,
            augmented_masks=aug_masks
        )
        val_ds = SegmentationDataset(val_paths, val_masks, transform=self.norm_transform)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def _compute_stats(self, train_paths, mask_paths):
        # only images for stats
        temp_ds = SegmentationDataset(train_paths, mask_paths, transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ]))
        loader = DataLoader(temp_ds, batch_size=self.batch_size, shuffle=False)

        # compute mean and std to normalize
        mean = torch.zeros(3)
        std = torch.zeros(3)
        total = 0

        # compute class(PE & background) counts for weight of weighted cross-entropy loss
        counts = torch.zeros(2, dtype=torch.float64)
        total_pixels = 0

        for imgs, masks, _ in loader:
            batch = imgs.size(0)
            total += batch
            mean += imgs.mean([0, 2, 3]) * batch
            std += imgs.std([0, 2, 3]) * batch
            
            # for weighted cross-entropy loss
            for c in range(2):  # class numbers
                counts[c] += (masks == c).sum().item()
            total_pixels += masks.numel()

        mean /= total
        std /= total

        weights = total_pixels / (counts * 2)

        logging.info(f"Computed stats - mean: {mean}, std: {std}\n\n")
        return mean.tolist(), std.tolist(), weights
    
class Trainer:
    def __init__(self, cfg: ConfigManager,ckpt_mgr: CheckpointManager, data_mgr: DataManager):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.model_name = cfg.model_name
        self.epochs = cfg.epochs
        self.patience = cfg.patience
        self.lr = cfg.learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.ckpt_mgr = ckpt_mgr
        self.data_mgr = data_mgr

    def train_fold(self, fold, train_loader, val_loader):
        model = self.cfg.build_model()
        model.to(self.device)
        criterion = self._get_loss_function()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        optimizer_name = optimizer.__class__.__name__
        scheduler_name = scheduler.__class__.__name__
        logging.info(f"Optimizer: {optimizer_name}")
        logging.info(f"Scheduler: {scheduler_name}\t Step Size: {scheduler.step_size}\t Gamma: {scheduler.gamma}\n\n")

        best_iou, best_acc, best_dice = 0, 0, 0
        patience = 0
        records = []

        for epoch in range(0, self.epochs):
            model.train()
            train_loss = 0
            for imgs, masks, _ in train_loader:
                imgs, masks = imgs.to(self.device), masks.long().to(self.device)    # CrossEntropyLoss 需要 Long 類型的標籤
                optimizer.zero_grad()
                if self.model_name == 'GLFR':
                    out = model(imgs)['main_out']
                else:
                    out = model(imgs)
                
                loss = criterion(out, masks.squeeze(1))
                # loss = criterion(out, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            val_metrics = self._evaluate(model, criterion, val_loader)
            print(
                f"Fold {fold} - Epoch [{epoch+1}/{self.epochs}], Average Train Loss: {train_loss:.4f}", 
                f"Val_loss: {val_metrics['loss']:.4f}, mean_dice: {val_metrics['dice']:.4f}, mean_iou: {val_metrics['iou']:.4f}, mean_accuracy: {val_metrics['acc']:.4f}"
            )
            logging.info(
                f"Fold {fold} - Epoch [{epoch+1}/{self.epochs}]: TrainLoss={train_loss:.4f}, "
                f"ValLoss={val_metrics['loss']:.4f}, "
                f"IoU={val_metrics['iou']:.4f}, Dice={val_metrics['dice']:.4f}, Acc={val_metrics['acc']:.4f}"
            )

            # early stopping & save checkpoint
            if val_metrics['iou'] > best_iou:
                best_iou, best_acc, best_dice = val_metrics['iou'], val_metrics['acc'], val_metrics['dice']
                patience = 0
                self.ckpt_mgr.save(
                    {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'normalization':self.data_mgr.norm_transform,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_iou': best_iou
                    }, fold
                )
                logging.info(f"Early stopping: {patience} / {self.patience}")
            else:
                patience += 1
                if patience >= self.patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
            scheduler.step()
            records.append(val_metrics)
        return best_dice, best_iou, best_acc, records

    def _evaluate(self, model, criterion, loader):
        model.eval()
        losses, dices, ious, accs = [], [], [], []
        with torch.no_grad():
            for imgs, masks, _ in loader:
                imgs, masks = imgs.to(self.device), masks.long().to(self.device)
                if self.model_name == 'GLFR':
                    out = model(imgs)['main_out']
                else:
                    out = model(imgs)     
                losses.append(criterion(out, masks.squeeze(1)).item())
                pred = torch.argmax(out, dim=1).cpu().numpy()
                true = masks.squeeze(1).cpu().numpy()
                pred_flat, true_flat = pred.flatten(), true.flatten()
                accs.append(accuracy_score(true_flat, pred_flat))
                ious.append(jaccard_score(true_flat, pred_flat, average='binary'))
                dices.append(f1_score(true_flat, pred_flat, average='binary'))
        return {
            'loss': np.mean(losses),
            'iou': np.mean(ious),
            'dice': np.mean(dices),
            'acc': np.mean(accs)
        }
    
    def _get_loss_function(self):
        if self.cfg.loss_function == 'weighted-ce':
            weight = self.data_mgr.weight.float().to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight)
            logging.info(f"Using Weighted Cross-Entropy Loss with weights: {weight}")
            return criterion

        elif self.cfg.loss_function == 'ce':
            ce_loss = nn.CrossEntropyLoss()
            logging.info("Using Cross-Entropy Loss")
            return ce_loss
        
        elif self.cfg.loss_function == 'focal':
            focal_loss = smp.losses.FocalLoss(
                mode='multiclass',
                alpha=0.25, 
                gamma=2.0, 
                reduction='mean'
            )
            logging.info("Using Focal Loss")
            return focal_loss
        
        elif self.cfg.loss_function == 'dice':
            dice_loss = smp.losses.DiceLoss(
                mode='multiclass',
                smooth=1e-6, 
            )
            logging.info("Using Dice Loss")
            return dice_loss
        elif self.cfg.loss_function == 'combo':
            # combo_loss = ComboLoss(alpha=0.5, beta=0.4, smooth=1.0)
            combo_loss = ComboLossWrapper(alpha=0.5, beta=0.4, smooth=1e-6)
            logging.info("Using Combo Loss")
            return combo_loss
        
        else:
            print("Using standard Cross-Entropy Loss")
            return nn.CrossEntropyLoss()

class ComboLossWrapper(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, reduction='mean'):
        super().__init__()
        self.combo = ComboLoss(alpha, beta, smooth, reduction)

    def forward(self, logits, targets):
        # logits: (N,2,H,W), targets: (N, H, W) label 0/1
        # 1. 取第 1 類 (前景) 的 logits
        fg_logits = logits[:, 1:2, :, :]         # shape (N,1,H,W)
        # 2. 把 targets 變成 float mask
        fg_mask = (targets == 1).float().unsqueeze(1)  # (N,1,H,W)
        return self.combo(fg_logits, fg_mask)

class ComboLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Dice 部分
        intersection = (probs * targets).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (
               probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth)
        dice_loss = 1 - dice

        # 加權 BCE 部分
        bce = - (self.beta * targets * torch.log(probs + 1e-7) +
                 (1 - self.beta) * (1 - targets) * torch.log(1 - probs + 1e-7))

        # 組合
        loss = self.alpha * bce.mean(dim=(2, 3)) + (1 - self.alpha) * dice_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
class CrossValidator:
    def __init__(self, cfg: ConfigManager):
        self.data_mgr = DataManager(cfg)
        self.ckpt_mgr = CheckpointManager(cfg.output_dir)
        self.trainer = Trainer(cfg, self.ckpt_mgr, self.data_mgr)
        self.num_folds = cfg.num_folds
        self.output_dir = cfg.output_dir
        self.output_csv = cfg.output_csv

    def run(self):
        full_ds, _ = prepare_data(self.data_mgr.image_dir, self.data_mgr.mask_dir)
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(full_ds), 1):
            logging.info(f"------- Training fold {fold} -------")
            train_loader, val_loader = self.data_mgr.prepare_fold(train_idx, val_idx)
            dice, iou, acc, _ = self.trainer.train_fold(fold, train_loader, val_loader)
            results.append({'fold': fold, 'dice': dice, 'iou': iou, 'accuracy': acc})

        df = pd.DataFrame(results)
        logging.info(f"Cross-validation results :")
        logging.info(f"{df}\n\n")
        print(df)
        summary = df[['dice', 'iou', 'accuracy']].agg(['mean', 'std'])
        print(summary)

        logging.info(f"Cross-validation summary:")
        logging.info(
            f"""Results of mean and std: 
            mean dice: {df['dice'].mean():.4f}\t std dice: {df['dice'].std():.4f}
            mean iou: {df['iou'].mean():.4f}\t std iou: {df['iou'].std():.4f} 
            mean accuracy: {df['accuracy'].mean():.4f}\t std accuracy: {df['accuracy'].std():.4f}"""
        )
        mean_std_fold_df = pd.DataFrame({
            "time": [self.trainer.cfg.timestamp],
            "mean_dice": [df['dice'].mean()],
            "std_dice": [df['dice'].std()],
            "mean_iou": [df['iou'].mean()],
            "std_iou": [df['iou'].std()],
            "mean_accuracy": [df['accuracy'].mean()],
            "std_accuracy": [df['accuracy'].std()],
        })

        logging.info(f'\n{mean_std_fold_df}')
        mean_std_fold_df.to_csv(self.output_csv, mode='a', header=True, index=False)


def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S %p"
    )


def main():
    # cfg = ConfigManager('Segmentation/config.yaml')

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='Segmentation/config.yaml', help="Path to the config file")
    args = parser.parse_args()

    config_path = args.config
    cfg = ConfigManager(config_path)

    setup_logging(cfg.log_path)
    cfg.write_config_to_log()

    cv = CrossValidator(cfg)
    cv.run()
    logging.info(f"Training completed !!")

if __name__ == '__main__':
    main()
