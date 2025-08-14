import os
import subprocess

extend_options = ["extend_05", "extend_10", "extend_15", "extend_20", 
                    "extend_25", "extend_30", "extend_35", "extend_40",
                    "extend_45", "extend_50", "extend_55", "extend_60",
                    "extend_65", "extend_70"]

folder_path = "Feature_extraction"
output_dir = os.path.join(folder_path, 'script_results')  # 保存結果的目錄
os.makedirs(output_dir, exist_ok=True)
train_script = "Feature_extraction/extract.py"  # 替換為你的 train.py 路徑（相對路徑）

for extend_option in extend_options:
    print(f"Running extract.py with roi {extend_option}")
    result_file = os.path.join(output_dir, f"result_{extend_option}.txt")

    with open(result_file, "w") as output_file:
        subprocess.run(
            ["python ", train_script, "--o", extend_option],  # 假設 train.py 支持 --config 參數
            stdout=output_file,
            stderr=subprocess.STDOUT
        )
    print('---- done. ----')