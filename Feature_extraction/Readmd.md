extract.py 用來提取影像的紋理特徵，執行時需輸入參數，選擇要提取特徵的影像類型 (eg. --o roi)

config.yaml 檔是原本的 pyradiomics clone 下來的東西，不要動他


utils 資料夾內的 create_mask.py 用於創造超音波扇形區域的 mask，因為提取需要原影像及 mask ，若要提取原始影像紋理特徵，就必須創建一個扇形區域的 mask，存為 full_selection_mask.jpg
43 及 67 影像大小不同，另外處理，儲存為 full_selection_mask_spe.jpg

scriptForExtendMaskRunExtract.py 用於自動提取不同擴展程度的影像特徵，標準輸出儲存於 script_results 資料夾中，方便檢視及 debug

輸出的所有 csv 檔案儲存於 output_csv 資料夾中