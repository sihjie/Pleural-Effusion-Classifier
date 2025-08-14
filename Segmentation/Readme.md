config.yaml:
    參數設定檔，可設定參數如下：
    - 分割模型
    - 幾折訓練
    - 是否 early stopping
    - epochs
    - batch size
    - learning rate
    - loss function
    - 用於分割之影像類型（是否做過前處理）
    - 是否 augmentation

train.py: 
    可不給 config 路徑，已有 default 設定
    根據 config default 設定檔訓練網路

train_config_runner.py:
    若要一次跑好幾種參數組合，請使用此程式碼
    執行時的標準輸出存於 script_results，方便檢視及 debug

data_loader.py:
    train.py 會呼叫，用於處理資料，包含資料擴增

all_predict.py:
    ！！！ 執行前注意 58 ckpt_dir = ... ！！！ 請選擇最佳結果之模型參數檔所在之資料夾
    並在 90 ckpt_path = ... 設定最佳 fold 之參數檔
    111 也需視最佳模型更改
    
    此程式碼會將所有輸出結果存於 predicted_masks 資料夾內，內含：
    - masks資料夾：儲存最佳模型參數組合之最佳 fold 預測出的所有 mask
    - each_pred_results.csv：最佳模型參數組合最佳 fold 預測每張影像之 Dice，IOU，accuracy
    - log_file.log：log 檔，儲存了最佳模型參數組合之各 fold 結果

overlay.py：
    輸出結果只 for 報告
    用於疊加 ground truth 及 預測 mask 在原圖上
