# Bone-density
利用opencv kmeans 對骨密度填充物進行影像分割(Image Segmentation)，透過二值化、開運算、高斯模糊、輪廓尋找對影像後處理，結合GUI達到方便快速分割目標物與背景，並計算分割出來的骨密度空洞像素面積值。
以下為分割的結果，像素面值會因相機距離與相機畫素有所不同，僅供參考：
![image](https://user-images.githubusercontent.com/82528634/134306212-a592938f-93ce-4a98-8e9b-1d5de606c707.png)
![image](https://user-images.githubusercontent.com/82528634/134306249-8b1c38d5-8d99-4209-ad66-085468c67cf6.png)
![image](https://user-images.githubusercontent.com/82528634/134306257-f7c01ff4-6cf1-4c19-9a3b-f86184711435.png)



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
                          
     本程式分為兩種模式，照片模式與拍照模式，拍照模式為後會進入框選需範圍。
     

     
     
                          


tryToGet_final.py 
----------------------------------------------------------



