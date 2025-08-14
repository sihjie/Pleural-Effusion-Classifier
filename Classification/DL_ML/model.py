# model.py

import torch
import torch.nn as nn
import torchvision.models as models

from Classification.DL_ML.config import ConfigManager


class PleuralEffusionClassifier(nn.Module):
    def __init__(self, config_manager: ConfigManager, num_classes=2):
        """
        使用預訓練的模型進行 full fine-tuning，並將最後一層改為 Identity 以便取得深度特徵，
        再接上自定義的分類器。
        """
        super(PleuralEffusionClassifier, self).__init__()
        self.model_name = config_manager.model_name
        # self.dropout = config_manager.dropout
        # self.dropout_rate = config_manager.dropout_rate
        
        # 1. ResNet50
        if self.model_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # 將最後的 fc 層替換成 Identity 以取出 2048 維特徵
            self.backbone.fc = nn.Identity()
            self.feature_dim = 2048

        # 2. VGG19
        elif self.model_name == "vgg19":
            self.backbone = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
            # 移除 VGG19 classifier 最後一層
            self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier.children())[:-1])
            self.feature_dim = 4096

        # 3. EfficientNet-V2-S
        elif self.model_name == "efficientnetv2_s":
            self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            self.backbone.classifier = nn.Identity()  # 去掉最後一層全連接層
            self.feature_dim = 1280

        # 4. ConvNeXt-Tiny
        elif self.model_name == "convnext_tiny":
            # 載入預訓練的 ConvNeXt-Tiny
            self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            # 移除預設的分類層，只保留到 Dropout 之前的特徵抽取部分
            self.backbone.classifier = nn.Sequential(
                *list(self.backbone.classifier.children())[:-1]
            )
            # ConvNeXt-Tiny 的特徵維度是 768
            self.feature_dim = 768

        # 5. MobileNetV3-Large
        elif self.model_name == "mobilenetv3_small":
            self.backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            )
            # 去掉最後一層 Linear
            self.backbone.classifier = nn.Sequential(
                *list(self.backbone.classifier.children())[:-1]
            )
            # classifier 第一層 Linear 的 output 大小就是 feature_dim
            self.feature_dim = self.backbone.classifier[0].out_features

        # 6. DenseNet-121
        elif self.model_name == "densenet121":
            self.backbone = models.densenet121(
                weights=models.DenseNet121_Weights.DEFAULT
            )
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.feature_dim = num_features

        else:
            raise ValueError(f"未知的模型名稱: {self.model_name}")

        # 定義分類器 (可在後續 concat 臨床資料後調整輸入維度)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.feature_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, num_classes)
        # )

        # if self.dropout:
        #     self.fc = nn.Sequential(
        #         nn.Linear(self.feature_dim, 256),
        #         nn.ReLU(),
        #         nn.Dropout(self.dropout_rate),
        #         nn.Linear(256, num_classes)
        #     )
        # else:
        #     self.fc = nn.Sequential(
        #         nn.Linear(self.feature_dim, 256),
        #         nn.ReLU(),
        #         nn.Linear(256, num_classes)
        #     )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),  # 更新輸入維度
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        回傳分類結果與深度特徵（例如取自預訓練網路最後一層前的向量）
        """
        deep_features = self.backbone(x)
        output = self.fc(deep_features)
        
        return output, deep_features  # 同時回傳分類輸出及深度特徵
