——————————————————————————————————————————————————
———— 此資料夾內程式碼用於使用機器學習訓練 CSV 檔 ————
——————————————————————————————————————————————————

config.yaml 為參數設定檔

train.py 會根據 config.yaml 來執行模型參數組合(grid search)做訓練，需輸入參數選擇 config 檔案
會吃到參數檔的 '模型類別'，'每個模型的設置參數'，'要 drop 的特徵'等
使用 grid search 遍歷每個模型的所有參數選擇，內建 5-fold cross validation，經驗證後可輸出最佳模型選擇
為了確保最佳參數選擇訓練時的穩定性，實作了外層 5-fold cross validation（內層為 gridsearch 內建的 5-fold）

其餘參數：'是否 SMOTE'，'是否 Tomek Link'，'input 的 csv 檔'

要遍歷其餘參數需執行 train_config_runner.py，使用三個 for 迴圈新建暫時的 config.yaml 檔並作為參數執行 train.py
下指令：python -m Classification.ML.train_config.runner

若要直接執行 train.py，請檢查 config.yaml 檔，確保要執行的參數組合
下指令：python -m Classification.ML.train --config Classification/ML/config.yaml

script_results 資料夾存 train_config_runner.py 執行時輸出的標準輸出，方便檢視及 debug


results：
依照 input type 做分類，也就是輸入的 csv 檔為各個資料夾名稱，內層資料夾由程式執行時的時間命名
資料夾內容如下：
|- confusion_matrix_fig // 存放各 model 的各 fold 的 confusion matrix
|- ROC_fig              // 存放各 model 的各 fold 的 roc curve
|- shap_fig             // 存放各 model 的各 fold 的 shap 圖還有 shap-summary 圖
|- {model name}_cv_results  // 模型的 gridsearch 內建的 5-fold 的各項指標的平均值及標準差
|- ML_training_logger.log  // 訓練 log 檔，內容含外層 5-fold 每個 fold 中最佳參數組合及內層 5-fold 的平均值及標準差

