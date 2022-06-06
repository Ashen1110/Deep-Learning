# Loss function

## binary_crossentropy

## dice_loss
dice loss 可以簡單的理解為兩個論括區域的相似程度，如下圖公式，A、B 表示兩個輪廓區域所包含的點集合
![image](https://user-images.githubusercontent.com/67892268/172189210-4aff2f4d-0c46-40b7-9342-3dfabc0d75d3.png)

程式碼：
``` py
def dsc(y_true, y_pred):
    smooth=1
    intersection = K.sum(y_true*y_pred, axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    score = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss
```
結論：
* 結果是 overfit
* 猜測因為我們的 Unet 架構使用 softmax，其梯度簡化而言為 p-t，其中 t 為目標值；p 為預測值。但是 dice loss的梯度為 $ \frac {2t^2}{(p+t)^2} $，如果 p 和 t 過小會造成梯度變化劇烈，導致訓練困難
* 可能比較適合場景不平衡的情況下，例如：前列腺 MRI 容積圖
* reference: https://arxiv.org/pdf/1606.04797.pdf

## binary_cross_entropy + dice_loss

## IOU
有點類似於 dice loss                                                                                               
![image](https://user-images.githubusercontent.com/67892268/172197969-9449e747-f392-4cad-ba6b-b19e2d574512.png)

其 loss function 定義為：$$ \frac {I(X)}{U(X)} $$ 其中 $ I(X) = X * Y $，$ U(X) = X+Y - I(X) $ 

X 為預測值，Y 為真實標籤值

程式碼：
``` py
def IOU(y_true, y_pred, eps=1e-6):
    if np.max(y_true)==0.0:
        return IOU(1-y_true, 1-y_pred)
    intersection = K.sum(y_true*y_pred, axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2]) - intersection
    return -K.mean((intersection+eps)/(union+eps), axis=0)
```
結論：
* 雖然 IOU 的好處是 loss 值具有非負性
* 但是會出現 loss 值是 nan 的問題，究其原因當背景和目標的 IOU 為 0 時，loss 為 Inf，網絡無法進行訓練

## tversky_loss
Tversky 係數是由 dice 係數和 Jaccard 係數的一種廣義係數，定義如下圖：
![image](https://user-images.githubusercontent.com/67892268/172200226-b8240865-d23a-4f10-b991-87b343fce6c7.png)

此時，A 為預測值，B 為真實標籤
當  $\alpha$  和 $ \beta $ 為 0.5 時，Tversky 係數 = dice 係數；$ \alpha $ 和 $ \beta $ 為 1 時，Tversky 係數 = Jaccard 係數
在 T(A, B) 中，$ |A - B| $ 代表假陽性；$ |B - A| $ 代表假陰性，透過調整 $\alpha$  和 $ \beta $ 分別控制兩者的權衡


程式碼：
``` py
def tversky(y_true, y_pred, smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos* y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1-tversky(y_true, y_pred)
```

結論：
* 此方法考慮到處理的數據類別極度不均衡的情況下的數據(原本該標記的卻沒標記、原本沒標的卻標了)，在犧牲一定精度的情況下，希望可以提高像素分類的 Recall 值 (預測為正例的真實正例佔所有真實正例的比例)
* 但是此方法看來還是沒能逃過黑色背景佔比太大的問題

## focal_loss + tversky_loss

## generalized_dice_loss
在使用 dice 時，對小目標是非常不利的。因為在只有前景和背景的情況下，小目標一旦有部份像素預測錯誤，就容易導致 Dice 大幅度的變動，導致梯度變化劇烈、訓練不穩定。
因此，generalized dice loss 將多個類別的 dice 進行整合，使用一個指標對分割結果進行量化。

其公式如下：                                                                                                
![image](https://user-images.githubusercontent.com/67892268/172205562-e4e5d9a4-53e1-4d5a-96fe-b0f7417c1787.png)
