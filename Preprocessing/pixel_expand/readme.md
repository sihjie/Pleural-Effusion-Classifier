此資料夾專門存放有關 pixel expand 相關之資料，結果，以及程式碼

以下介紹各層級檔案及資料夾：

pixel_expand
|- inter_window_mask 
|    |- inter_mask_expand_##    // window 內 延伸 ## pixels 的 mask 影像
|    
|- origin_expand_mask
|    |- roi_expand_##    // 存放延伸 ## pixel 的 mask 影像
|
|- full_selection_mask.jpg  // window 大小的 mask，用於限制 pixel expand 範圍
|- full_selection_mask_spe.jpg  // window大小的 mask，用於'0043'及'0067'影像（兩張超音波影像大小及解析度與其他不同，另外處理）
|
|- mask_expansion.py    // 將分割出的 predicted mask 做不同程度的(5-95) pixel expansion，並分別將結果存至 origin_expand_mask/mask_expand_##
|- intersection.py  // 取得 window 內 pixel expansion 的 mask 影像，結果存在 inter_window_mask/inter_mask_expand_##