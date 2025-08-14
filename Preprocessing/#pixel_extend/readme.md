此資料夾專門存放有關 pixel extended 相關之資料，結果，以及程式碼

以下介紹各層級檔案及資料夾：

pixel_extend
|- intersection         // 存放有關'將 pixel extension 限制在超音波 window 內'的相關程式碼及結果
    |- inter_roi        // window 內 pixel extension 的 RoI 們（純報告用）
    |- overlay_img      // window 內 pixel extension 疊加在原圖的影像們（純報告用）
    |- roi_extend_##    // window 內 延伸 ## pixel 的 mask 影像
    |- full_selection_mask.jpg  // window 大小的 mask，用於限制 pixel extend 範圍
    |- full_selection_mask_spe.jpg  // window大小的 mask，用於'0043'及'0067'影像（兩張超音波影像大小及解析度與其他不同，另外處理）
    |- get_roi.py       // 取得 window 內 pixel extension 的 RoI 們， 結果存在 inter_roi
    |- intersection.py  // 取得 window 內 pixel extension 的 mask 影像，結果存在 roi_extend_##
    |- overlay.py       // 取得 window 內 pixel extension mask 疊加在原圖上的影像，結果存在 overlay_img
|- origin_extend_mask
    |- overlay_img      // 存放不同程度的 pixel extension mask 疊加在原圖上之影像（純報告用）
    |- roi_extend_##    // 存放延伸 ## pixel 的 mask 影像
    |- overlay.py       // 將 pixel extension mask 疊加在原圖上之程式，結果存在 overlay_img 
|- roi_extension.py     // 將分割出的 predicted mask 做不同程度的(5-95) pixel extension，並分別將結果存至 origin_extend_mask 資料夾內的 roi_extend_## 資料夾內