——————————————————————————————————————————————————
———— 此資料夾內程式碼用於使用深度網路訓練整張影像 ————
——————————————————————————————————————————————————

train.py 中 import data_preparation.py、 model.py 及 loss_function.py 三個檔案，分別用於：
|   × data_preparation.py：處理資料
|   × model.py：定義深度學習模型
|   × loss_function.py：定義 loss function

！！！執行前記得更改檢查 config.yaml 檔案 ！！！

train.py 預設會根據 "run_all" 設定決定是否將 config.yaml 檔中各個設定的 options 值全部執行，若只要執行 default 值，可將 run_all 的值改為 False
if __name__ == "__main__":
    main(run_all=True) 

若 "run_all" 設定為 True，會在 results 資料夾內生成 log 檔 "{time_stamp}_all_parameter_combination.log"，方便比較各參數組合結果

train.py 會一次執行完 5-fold cross validation training，並使用 testing dataset 進行 5-fold 測試，最後選用 5-fold test 中最好的結果之 model path 來做所有資料的prediction

輸出結果全部存在 results 資料夾中
results
|   資料夾命名規則：{輸入影像} --{深度網路} eg.expand_15 -- resnet50
|    |- 內層資料夾命名規則：{time_stamp}_{[loss_function]}_{lr}_aug_{if aug}
|    |   |- accuracy_curve_{fold}.png   // 每個 fold 的 accuracy curve
|    |   |- checkpoint_fold_{fold}.pth  // 每個 fold 的模型參數檔
|    |   |- confusion_matrix_{fold}.png // 每個 fold 的 confusion matrix
|    |   |- loss_curve_{fold}.png       // 每個 fold 的 loss curve
|    |   |- training.log                // 此參數組合 5-fold training 的 log 檔
|    |
|    |- 5-fold_predict_all.csv  // 同樣 {輸入影像} 搭配 {深度網路} 組合之所有 predict 結果
|    |- 5-fold_testing_inference.csv    // 同樣 {輸入影像} 搭配 {深度網路} 組合之所有 test 結果
|    |- 5-fold_training.csv    // 同樣 {輸入影像} 搭配 {深度網路} 組合之所有 training 結果

× train：5-fold training set
× test： testing set
× predict： all data

<!-- 舊結果
用輸入影像類別做分類，再用 model 做分類
eg.
    results
    |- CLAHE
    |    |- resnet50
    |    |- vgg19
    |- original
    |    |- resnet50
    |    |- vgg19
    |- RoI
    |    |-resnet50
    |    |- vgg19
    |- x-ray
    |    |- resnet50
    |- all_parameter_combinations.log


結果檔案說明：
all_parameter_combinations.log  ：儲存 5-fold training 的結果，儲存各參數組合，及組合的 5-fold 結果

每個 model 資料夾內會有以參數組合命名之資料夾，內有該組合 training 的結果，包含：
一份 training.log，以下皆有五份（因為 5-fold）：accuracy_curve 圖檔，checkpoint.pth， confusion matrix 圖檔，loss curve圖檔
並且有 test_inference 資料夾存放 test 資料及 predict_all 資料夾存放 predict 所有資料的結果

test_inference 內有 5 張 confusion matrix 圖檔，一份 inference log 檔

predict_all 內有 1 張 confusion matrix 圖檔，一份 predict_all log 檔，及一份所有資料預測結果的 csv 檔

查找最佳結果的方法：
model 內除了各參數命名之資料夾外，還各有一份 5-fold_training.csv, 5-fold_testing_inference.csv, 5-fold_predict_all.csv
通常會看 testing_inference 的最佳結果，找到後對照 timestamp 找到對應資料夾可以知道該次訓練所用參數及各項結果 -->


