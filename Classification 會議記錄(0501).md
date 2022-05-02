# Classification 會議記錄(05/01)

## Input 

2d-image

## 架構

YOLO.v4 / unet

## 發現問題

出血點太不明顯以至於無法標記

## Now working

解決方法 : 

1. 對影片資料做預處理，重新跑一次yolo
2. 對影片資料做預處理，換 Model - unet

## 下次開會時間

5/7 (六) 22:00