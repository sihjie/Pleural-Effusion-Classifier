import yaml
import subprocess
import os

# 定義 YAML 文件和 train.py 的路徑
folder_path = "Segmentation"
yaml_path = os.path.join(folder_path, 'config.yaml')  # 替換為你的 YAML 文件路徑
train_script = "Segmentation.train"  # 替換為你的 train.py 路徑（相對路徑）
output_dir = os.path.join(folder_path, 'script_results')  # 保存結果的目錄
os.makedirs(output_dir, exist_ok=True)

# 讀取 YAML 文件
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

# 獲取 [data][input][options]
# model_options = config['model_setting']['model_name']['options']
# image_options = config['data_setting']['img_type']['options']
loss_options = config['model_setting']['loss_function']['options']
batch_size_options = config['model_setting']['batch_size']['options']
model_type = config['model_setting']['model_name']['default']
# 依次替換 [data][input][default] 並執行 train.py
for loss_type in loss_options:
    config['model_setting']['loss_function']['default'] = loss_type
    print(f"Running train.py with model = {model_type}")
    print(f"Running train.py with loss function = {loss_type}")

    # 保存修改後的 YAML 文件（臨時）
    temp_yaml_path = f"{folder_path}/temp.yaml"
    with open(temp_yaml_path, "w") as temp_file:
        yaml.dump(config, temp_file)
    
    # 執行 train.py 並傳遞配置文件
    result_file = os.path.join(output_dir, f"result_{model_type}___{loss_type}.txt")
    with open(result_file, "w") as output_file:
        subprocess.run(
            ["python", "-m", train_script, "--config", temp_yaml_path],  # 假設 train.py 支持 --config 參數
            stdout=output_file,
            stderr=subprocess.STDOUT, 
            check=True
        )
    print('---- run train.py one time done. ---- \n')
    # 可選：刪除臨時 YAML 文件
    os.remove(temp_yaml_path)

print("All configurations have been processed.")
