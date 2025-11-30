pixel_expand 自己有一個 Readme

Preprocessing

| 資料夾 | 說明 |
| ------------- | ------------- |
| BM3D  | 放經過 BM3D 的影像  |
| BM3D_CLAHE  | 放經過 BM3D 再經過 CLAHE 處理的影像  |
| CLAHE        | 放經過 CLAHE 處理過的影像 |
| Cropped      | 放 delabeled 且 cropped 後的影像 |
| LabelsCropped    | 放僅 cropped 後的影像（報告用） |
| Mask         | 放從 label 影像得到的 mask 影像 |
| pixel_expand | 跟 pixel_expand 有關的都放裡面 |

| 檔案 | 說明 |
| ------------- | ------------- |
| clahe.py     | 用於將影像做 CLAHE 影像增強 |
| delabeled_crop.py    | 將影像 delabeled 及裁切 |
| mask.py      | 取得 mask ground truth |
| get_roi.py   | 取得 ROI 影像 |
| polygon_points.pptx  | 用 ppt 手動計算 cropped 出扇形面積時需要的多邊形的坐標 |

- **除 BM3D_allstages，其餘 BM3D 皆只經過第一階段去雜訊（Basic Estimation）**
