# 這個檔案用來定義你要用的深度學習模型。這裡使用預訓練的 ResNet 進行影像分類。

import torch
import torch.nn as nn
import torchvision.models as models

class PleuralEffusionClassifier(nn.Module):
    def __init__(self, model_name, num_classes, clinical_features_dim):
        super(PleuralEffusionClassifier, self).__init__()

        self.model_name = model_name

        if model_name == "resnet50":
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.resnet.fc = nn.Identity()  # 去掉最後一層全連接層
            self.feature_dim = 2048  # ResNet50 的特徵維度

        elif model_name == "vgg19":
            self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
            self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1]) # 去掉最後一層
            self.feature_dim = 4096 # VGG19 的特徵維度
        
        elif model_name == "googlenet":
            self.googlenet = models.googlenet(pretrained=True, aux_logits=True)
            self.googlenet.fc = nn.Identity()  # 去掉最後一層全連接層
            self.feature_dim = 1024  # GoogLeNet 的特徵維度

        #######################################################################
        # 臨床數據處理的全連接層
        self.fc_clinical = nn.Sequential(
            nn.Linear(clinical_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # 融合後進行分類的全連接層
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim + 128, 256),  # 更新輸入維度
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, clinical_data):
        # 根據 model_name 選擇不同的模型進行前向傳播
        if self.model_name == "resnet50":
            image_features = self.resnet(x)  # 輸出為 [batch_size, 4096]
        
        elif self.model_name == "vgg19":
            image_features = self.vgg(x)  # 輸出為 [batch_size, 4096]
                
        elif self.model_name == "googlenet":
            image_features = self.googlenet(x)
        #     # 使用 GoogLeNet 且 aux_logits 設為 True，返回三個值
        #     if self.googlenet.training and self.googlenet.aux_logits:
        #         x, aux1, aux2 = self.googlenet(x)
        #         clinical_features = self.fc_clinical(clinical_data)
        #         combined_features = torch.cat((x, clinical_features), dim=1)
        #         x = self.fc(combined_features)

        #         return x, aux1, aux2
        #     else:
        #         return self.googlenet(x)
        else:
            raise ValueError(f"未知的模型名稱：{self.model_name}")
        
        clinical_features = self.fc_clinical(clinical_data)
        combined_features = torch.cat((image_features, clinical_features), dim=1)  # [batch_size, feature_dim + 128]
        output = self.fc(combined_features)

        return output