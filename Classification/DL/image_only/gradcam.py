import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from model import PleuralEffusionClassifier  # Import the model class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義Grad-CAM類別
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None
        self.register_hooks()

    def register_hooks(self):
        # 註冊forward hook抓取activation，backward hook抓取gradient
        def forward_hook(module, input, output):
            self.activation = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x):
        # 前向傳播
        logits = self.model(x)
        pred_class = logits.argmax(dim=1).item()

        # 反向傳播，僅計算預測類別的梯度
        self.model.zero_grad()
        logits[0, pred_class].backward(retain_graph=True)

        # 計算Grad-CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activation[0]
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap, pred_class

# # Load the image and apply the transform
# def load_image(image_path):
#     image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     return image

# # Load the model and weights
# def load_model(model_path, model_name):
#     model = PleuralEffusionClassifier(model_name=model_name, num_classes=2)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()  # Set the model to evaluation mode
#     return model

# # Predict the class of the input image
# def predict(image, model):
#     with torch.no_grad():
#         image = image.to(device)
#         output = model(image)
#         _, predicted = torch.max(output, 1)  # Get the index of the highest score
#         return predicted.item()

# Main function to run the prediction
def main():
    # Define argparse for command line arguments
    parser = argparse.ArgumentParser(description="Predict the class of an input image using a pre-trained model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    # parser.add_argument('--model', type=str, default='vgg16', choices=['vgg16', 'vgg19', 'resnet18', 'resnet50'],
    #                     help="Choose the model architecture: 'vgg16', 'vgg19', 'resnet18', or 'resnet50'")
    args = parser.parse_args()

    # Check if GPU is available and set the device
    model = PleuralEffusionClassifier('resnet50', num_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 選擇Grad-CAM目標層
    gradcam = GradCAM(model.resnet, model.resnet.layer4[2].conv3)

    # Define the transform for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1504, 0.1502, 0.1500], std=[0.1810, 0.1810, 0.1810])  # Standard normalization for pretrained models
    ])

    image = Image.open(args.image_path).convert("RGB")
    original_width, original_height = image.size  # 取得原始圖像大小
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 推理並生成Grad-CAM熱力圖
    heatmap, pred_class = gradcam(input_tensor)

    # 將heatmap疊加到原圖上
    heatmap = cv2.resize(heatmap, (original_width, original_height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

    # 顯示和儲存可視化結果
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title(f"Prediction: {'Malignant' if pred_class == 1 else 'Benign'}")
    plt.show()
    output_path = 'Classification/DL/image_only/gradcam_result.png'
    cv2.imwrite(output_path, superimposed_img)
    print(f"Grad-CAM visualization saved at {output_path}")
    

if __name__ == "__main__":
    main()
