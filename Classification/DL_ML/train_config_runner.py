import yaml
import subprocess
import os

# 定義 YAML 文件和 train.py 的路徑
folder_path = "Classification/DL_ML"
yaml_path = os.path.join(folder_path, 'config.yaml')  # 替換為你的 YAML 文件路徑
train_script = "Classification.DL_ML.train_by_pth"  # 替換為你的 train.py 路徑（相對路徑）
output_dir = os.path.join(folder_path, 'script_results')  # 保存結果的目錄
os.makedirs(output_dir, exist_ok=True)

# 讀取 YAML 文件
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

# 獲取 [data][input][options]
image_options = config['data_setting']['image_type']['options']
feature_options = config['data_setting']['feature_type']['options']
dimred_options = config['dimred']['options']

# 依次替換 [data][input][default] 並執行 train.py
for image_type in image_options:
    print(f"Running train.py with [data_setting][image_type][default] = {image_type}")
    config["data_setting"]["image_type"]["default"] = image_type
    for feature_type in feature_options:
        print(f"Running train.py with [data_setting][feature_type][default] = {feature_type}")
        config["data_setting"]["feature_type"]["default"] = feature_type
        for dimred_option in dimred_options:
            print(f"Running train.py with [dimred][default] = {dimred_option}")
            config["dimred"]["default"] = dimred_option
            # 保存修改後的 YAML 文件（臨時）
            temp_yaml_path = f"{folder_path}/temp.yaml"
            with open(temp_yaml_path, "w") as temp_file:
                yaml.dump(config, temp_file)
            
            # 執行 train.py 並傳遞配置文件
            result_file = os.path.join(output_dir, f"result_{image_type}_{feature_type}.txt")
            with open(result_file, "w") as output_file:
                subprocess.run(
                    ["python", "-m", train_script, "--config", temp_yaml_path],  # 假設 train.py 支持 --config 參數
                    stdout=output_file,
                    stderr=subprocess.STDOUT, 
                    check=True
                )
        print('---- run train.py one time done. ---- ')
        # 可選：刪除臨時 YAML 文件
        os.remove(temp_yaml_path)

print("All configurations have been processed.")
