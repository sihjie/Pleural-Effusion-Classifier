import os
import logging

def setup_log(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # 確保日誌目錄存在

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)  # 清除所有舊的 handlers

    logging.basicConfig(
        filename=filename,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger('shap').setLevel(logging.WARNING)  # 禁用 SHAP 的 INFO 級別日誌
    print(f"Log file created at {filename}")