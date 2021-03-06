# Bone-density
利用 K-means 對骨植入填充物進行影像分割(Image Segmentation)，透過二值化、開運算、高斯模糊、輪廓尋找對影像後處理，結合GUI達到方便快速分割目標物與背景，並計算分割出來的骨密度空洞像素面積值。
以下為分割的結果，像素面值會因相機距離與相機畫素有所不同，僅供參考：

![image](https://user-images.githubusercontent.com/82528634/138313180-cc381269-32fb-4583-89af-dae71e725e75.png)

![image](https://user-images.githubusercontent.com/82528634/134306212-a592938f-93ce-4a98-8e9b-1d5de606c707.png)
Square Array
![image](https://user-images.githubusercontent.com/82528634/134306249-8b1c38d5-8d99-4209-ad66-085468c67cf6.png)
Random
![image](https://user-images.githubusercontent.com/82528634/134306257-f7c01ff4-6cf1-4c19-9a3b-f86184711435.png)
Gyroid



required
---------------------------------------------------------------------
    opencv-python       4.4.0.44
    numpy               1.19.5
    Pillow              5.3.0
    scipy               1.5.4
    
    
usage
--------------------------------------------------------
     Bone-density資料夾中，analysis 資料夾為分析後的詳細數據
                          num 資料夾為後處理的預設參數
                          pic 資料夾為最後分割完的影像
     Kmean的最佳參數設定為分2類訓練100次
    
                     
                          
     本程式分為兩種模式，照片模式與拍照模式，拍照模式為後會進入框選需範圍。


主程式為 tryToGet_final.py

以下為範例：
拍照完的選取需求位置
![image](https://user-images.githubusercontent.com/82528634/134316046-2f470e1a-0046-4e5b-bc80-1acfdd31ff33.png)
整體程式呈現
![image](https://user-images.githubusercontent.com/82528634/134316025-728c5954-a1f6-47c5-be28-ee7f64a75772.png)
     

目前影像實例分割部分已經完成
     




