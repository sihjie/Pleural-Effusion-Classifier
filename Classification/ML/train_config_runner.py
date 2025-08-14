import yaml
import subprocess
import os

# 定義 YAML 文件和 train.py 的路徑
folder_path = "Classification/ML"
yaml_path = os.path.join(folder_path, 'config.yaml')  # 替換為你的 YAML 文件路徑
train_script = "Classification/ML/train.py"  # 替換為你的 train.py 路徑（相對路徑）
output_dir = os.path.join(folder_path, 'script_results')  # 保存結果的目錄

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)

# 讀取 YAML 文件
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

# 獲取 [data][input][options]
input_options = config["data"]["input"]["options"]
smote_options = config["data"]["SMOTE"]["options"]
tl_options = config["data"]["TomekLinks"]["options"]

# 依次替換 [data][input][default] 並執行 train.py
for input_option in input_options:
    print(f"Running train.py with [data][input][default] = {input_option}")
    config["data"]["input"]["default"] = input_option
    for smote_option in smote_options:
        print(f"Running train.py with [data][SMOTE][default] = {smote_option}")
        config["data"]["SMOTE"]["default"] = smote_option
        for tl_option in tl_options:
            print(f"Running train.py with [data][TomekLinks][default] = {tl_option}")
            config["data"]["TomekLinks"]["default"] = tl_option

            # 保存修改後的 YAML 文件（臨時）
            temp_yaml_path = f"{folder_path}/temp.yaml"
            with open(temp_yaml_path, "w") as temp_file:
                yaml.dump(config, temp_file)
            
            # 執行 train.py 並傳遞配置文件
            result_file = os.path.join(output_dir, f"result_{input_option}_smote-{smote_option}_tl-{tl_option}.txt")
            with open(result_file, "w") as output_file:
                subprocess.run(
                    ["python ", train_script, "--config", temp_yaml_path],  # 假設 train.py 支持 --config 參數
                    stdout=output_file,
                    stderr=subprocess.STDOUT
                )
            print('---- run train.py one time done. ---- ')
            # 可選：刪除臨時 YAML 文件
            os.remove(temp_yaml_path)

print("All configurations have been processed.")
