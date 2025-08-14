import torch
import torch.nn as nn
import torchvision.models as models
import timm

class PleuralEffusionClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2):
        super(PleuralEffusionClassifier, self).__init__()

        self.model_name = model_name

        # 1. ResNet50
        if model_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone.fc = nn.Identity()  # 去掉最後一層全連接層
            self.feature_dim = 2048  # ResNet50 的特徵維度
        
        # 2. VGG19
        elif model_name == "vgg19":
            self.backbone = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
            self.backbone.classifier = nn.Sequential(
                            *list(self.backbone.classifier.children())[:-1]
                        )            
            self.feature_dim = 4096

        # 3. EfficientNet-V2-S
        elif model_name == "efficientnetv2_s":
            self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            self.backbone.classifier = nn.Identity()  # 去掉最後一層全連接層
            self.feature_dim = 1280

        # 4. ConvNeXt-Tiny
        elif model_name == "convnext_tiny":
            # 載入預訓練的 ConvNeXt-Tiny
            self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            # 移除預設的分類層，只保留到 Dropout 之前的特徵抽取部分
            self.backbone.classifier = nn.Sequential(
                *list(self.backbone.classifier.children())[:-1]
            )
            # ConvNeXt-Tiny 的特徵維度是 768
            self.feature_dim = 768

        # 5. MobileNetV3-Small
        elif model_name == "mobilenetv3_small":
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
        elif model_name == "densenet121":
            self.backbone = models.densenet121(
                weights=models.DenseNet121_Weights.DEFAULT
            )
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.feature_dim = num_features


        #######################################################################

        # 融合後進行分類的全連接層
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),  # 更新輸入維度
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        image_features = self.backbone(x)  # 輸出為 [batch_size, 2048]
        output = self.fc(image_features)

        return output