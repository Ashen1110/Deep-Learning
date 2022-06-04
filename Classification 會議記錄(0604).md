# Classification 會議記錄(06/04)

## Input

2d-image

## 架構

unet

## 發現問題

unet : 還是無法找到出血點 (overwhealming ) ，嘗試減少 overwhealming 的問題，但是效果不大。[有可能是測試資料本來就沒有出血點]

unet : 計算 loss 的方式(黑色點對的 % 數以及白色點對的 % 數)，可能會因為黑色點比例太多導致無法確實顯示白色點對的 % 數，即使預測出全黑的圖片準確率還是很高。(已改變 loss function 但是還是沒差)

## Now working

解決方法 :

1. 修改 unet 的 modle ，將 RGB 矩陣內的數值(255, 255, 255)改掉(改成16進位的表示法) [ overwhealming 更嚴重了]
2. 改變 matrix 的轉換方式 [ overwhealming 更嚴重了]
3. 試著只計算白色區域的 loss func. (只計算預測白色 pixel 跟實際白色 pixel 就好) [還是沒有嗚嗚]
4. 改變訓練資料的比例 [沒做在ㄍㄧㄥ一下]

