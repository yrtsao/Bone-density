from tkinter.constants import LEFT, RIGHT
from PIL.Image import NONE
from scipy.stats import mode
import cv2
import numpy as np
import csv
import time
import tkinter as tk
import tkinter.filedialog
import scipy.io as scio
import os
#--------------------------------------------------------------------主UI------------------------------------------------------------

window = tk.Tk()
window.title('Bone Density')
window.geometry('450x235')
label = tk.Label(window,text = '開始測試' ,font = ('Arial',12),width = 30)
label.pack()
label4 = tk.Label(window,text = '功能按鍵:' ,font = ('Arial',12),width = 30)
label4.pack()
label1 = tk.Label(window,text = '按(w) 顯示各階段影像處理成果' ,font = ('Arial',12),width = 30)
label1.pack()
label1 = tk.Label(window,text = '按(t) 暫存調整數值' ,font = ('Arial',12),width = 30)
label1.pack()
# label2 = tk.Label(window,text = '按(s) 顯示面積比例' ,font = ('Arial',12),width = 30)
# label2.pack()
label2 = tk.Label(window,text = '按(a) 計算框選空洞面積' ,font = ('Arial',12),width = 30)
label2.pack()
label3 = tk.Label(window,text = '按(d) 計算框選空洞內接圓直徑' ,font = ('Arial',12),width = 30)
label3.pack()
label4 = tk.Label(window,text = '按(q) 關閉離開' ,font = ('Arial',12),width = 30)
label4.pack()


avg = None
img = 0
xy_list = []
contours = []
contours_list = []
area_color = []
text_list = []
num_filename = './num/bone_density_log.npy'
Num_set = np.array(0)
lsPointsChoose = []
tpPointsChoose = []
pointsCount = 0
count = 0
pointsMax = 6
T = time.localtime()
cap = cv2.VideoCapture(1)  #相機設定
# cap.set(15,-1)
# cap.set(3, 960)
# cap.set(4, 960)


#---------------------------------------------------------------影像前處理------------------------------------------------------




def on_mouse(event, x, y, flags, param):  #紀錄滑鼠點擊位置並連成路徑
    # print("000000")
    
    global point1, point2, count, pointsMax
    # print("imgggg",img)
    global lsPointsChoose, tpPointsChoose  # 存入選擇的點
    global pointsCount  # 對鼠標按下的點計數
    global img,img2, ROI_bymouse_flag

    # 此行代碼保證每次都重新再原圖畫  避免畫多了
    cv2.imshow('frame0', img2)

    # -----------------------------------------------------------
    #    count=count+1
    #    print("callback_count",count)
    # --------------------------------------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:
    # cv2.EVENT_LBUTTONDOWN:  # 左鍵點擊
    # cv2.EVENT_FLAG_CTRLKEY # 拖曳 但還沒寫
        pointsCount = pointsCount + 1

        # if (pointsCount == pointsMax + 1):
        #     pointsCount = 0
        #     tpPointsChoose = []
        print('pointsCount:', pointsCount)
        point1 = (x, y)
        print (x, y)
        # 畫出點擊的點
        cv2.circle(img2, point1, 5, (0, 255, 0), 2)

        # 將選取的點保存到list列表裏
        lsPointsChoose.append([x, y])  # 用於轉化爲darry 提取多邊形ROI
        tpPointsChoose.append((x, y))  # 用於畫點
        # ----------------------------------------------------------------------
        # 將鼠標選的點用直線連起來
        print(len(tpPointsChoose))
        for i in range(len(tpPointsChoose)-1) :
            print('i', i)
            cv2.line(img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
            cv2.circle(img2, tpPointsChoose[i], 5, (0, 255, 0), 2)
        # ----------------------------------------------------------------------
        # ----------點擊到pointMax時可以提取去繪圖----------------
        

        cv2.imshow('frame0', img2)
        # main_function(file_path_roi)

    # -------------------------中鍵結束-----------------------------
    if event==cv2.EVENT_MBUTTONDOWN:  # 中鍵點擊
        print("middle-mouse")
        ROI_byMouse()
        ROI_bymouse_flag = 1

        pointsCount = 0
        tpPointsChoose = []
        lsPointsChoose = []


    # -------------------------右鍵按下清除軌跡-----------------------------
    if event == cv2.EVENT_RBUTTONDOWN:  # 右鍵點擊
        print("right-mouse")
        pointsCount = 0
        tpPointsChoose = []
        lsPointsChoose = []
        

def ROI_byMouse():  #將 on_mouse 記錄下的點連成多邊形
    global src, ROI, ROI_flag, mask2
    mask = np.zeros(img2.shape, np.uint8)
    # cv2.imshow('mask',mask)
    pts = np.array([lsPointsChoose], np.int32)  # pts是多邊形的頂點列表（頂點集）
    pts = pts.reshape((-1, 1, 2))
    # 這裏 reshape 的第一個參數爲-1, 表明這一維的長度是根據後面的維度的計算出來的。
    # OpenCV中需要先將多邊形的頂點座標變成頂點數×1×2維的矩陣，再來繪製

    # --------------畫多邊形---------------------
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
    ##-------------填充多邊形---------------------
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
    # cv2.imshow('mask', mask2)
    cv2.imwrite('mask.bmp', mask2)
    ROI = cv2.bitwise_and(mask2, img)
    cv2.imwrite('ROI.bmp', ROI)
    # cv2.imshow('ROI', ROI)
    main_function(ROI)

def ROI_capture():  #擷取 ROI_byMouse 多邊形內的影像
    global img2,img
    TT0 = time.localtime()
    while(True):
        # 從攝影機擷取一張影像
        ret, img = cap.read()

        # 顯示圖片
        cv2.imshow('frame', img)

        # 若按下 q 鍵則離開迴圈
        if cv2.waitKey(1) & 0xFF == ord('r'):
            file_path_roi = './pic/_ROI'+str(TT0)+'.jpg'
            cv2.imwrite(file_path_roi, img)
            # print(img)
            img2 = img.copy()
            cv2.namedWindow('frame0')
            cv2.setMouseCallback('frame0', on_mouse)
            
            break


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype = np.uint8),-1)
    return  cv_img

def knn(img):  #k-近鄰演算法，
    rows, cols = img.shape[:]
    dataNew = './dataNew.mat'
    # 影像二維像素轉換為一維
    data = img.reshape((rows * cols, 1))
    data = np.float32(data)

    # 定義中心 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 設置標籤
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness8, labels8, centers8 = cv2.kmeans(data, 2, None, criteria, 100, flags)
    img = labels8.reshape((img.shape))
    img = np.array(img, dtype = np.uint8)
    # scio.savemat(dataNew, {'img':img})
    img = img*128
    cv2.imwrite('knn.bmp', img)
    return img

def binarization(img,threshold1,threshold2,binary_type):
    binary = [cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV,cv2.THRESH_TRUNC,cv2.THRESH_TOZERO,cv2.THRESH_TOZERO_INV]
    ret, img = cv2.threshold(img, threshold1, threshold2,  binary[binary_type] )  # 二值化
    return img

def erode(img,kernal_size1):

    kernel1 = np.ones((kernal_size1,kernal_size1), np.uint8)
    img = cv2.erode(img, kernel1, iterations = 1) #侵蝕處理
    return img

def dilate(img,kernal_size2):

    kernel2 = np.ones((kernal_size2,kernal_size2), np.uint8)
    img = cv2.dilate(img, kernel2, iterations = 1) #膨脹處理
    return img

def blur(img,kernal_size3):

    # print(kernal_size3)
    # img = cv2.blur(img, (kernal_size3,kernal_size3)) #均值模糊
    # cv2.imwrite('blur.bmp',img)
    img = cv2.GaussianBlur(img, (kernal_size3,kernal_size3),0) #高斯模糊
    cv2.imwrite('GaussianBlur.bmp',img)
    # img = cv2.medianBlur(img, kernal_size3) #中值模糊
    # cv2.imwrite('medianBlur.bmp',img)
    # img = cv2.bilateralFilter(img,9,kernal_size3,kernal_size3) #雙邊模糊
    # cv2.imwrite('bilateralFilter.bmp',img)
    return img

def draw(img,img0,contours):
    # print("draw")    
    raw_dist = np.empty(img0.shape, dtype=np.float32)
    # print(raw_dist.shape)    
    for i in range(img0.shape[0]):
        for j in range(img0.shape[1]):
            raw_dist[i,j] = cv2.pointPolygonTest(contours, (j,i), True)

    # print(raw_dist.shape)    
    minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
    minVal = abs(minVal)
    maxVal = abs(maxVal)
    # print(minVal,maxVal)
    text1 = str(np.int(maxVal)*2)
    text_list.append(text1)
    cv2.circle(img,maxDistPt, np.int(maxVal),(255,255,255), 1, cv2.LINE_8, 0)
    #cv2.putText(img, text1, maxDistPt , cv2.FONT_HERSHEY_COMPLEX, 0.5, (22,245,255), 1, cv2.LINE_AA)

    return img

# def draw_area(img,img0,contours,contours_area):
#     # print("draw")    
#     raw_dist = np.empty(img0.shape, dtype=np.float32)
#     # print(raw_dist.shape)    
#     for i in range(img0.shape[0]):
#         for j in range(img0.shape[1]):
#             raw_dist[i,j] = cv2.pointPolygonTest(contours, (j,i), True)

#     # print(raw_dist.shape)    
#     minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
#     minVal = abs(minVal)
#     maxVal = abs(maxVal)
#     # print(minVal,maxVal)
#     text1 = str(np.int(contours_area))
#     text_list.append(text1)
    # cv2.circle(img,maxDistPt, np.int(maxVal),(255,255,255), 1, cv2.LINE_8, 0)
    # cv2.putText(img, text1, maxDistPt , cv2.FONT_HERSHEY_COMPLEX, 0.5, (22,245,255), 1, cv2.LINE_AA)

    # return img


def text_print1(list_test):
    T0 = time.localtime()
    dir_csv_path = './analysis'
    if not os.path.isdir(dir_csv_path):
        os.mkdir(dir_csv_path)
    print('CSV saving...')
    file_csv = time.strftime("%Y%m%d_%H%M%S", T0)
    count_text_list = {}
    if list_test !=[]:
        for i in list_test:
            if list_test.count(i) >= 1:
                count_text_list[i] = list_test.count(i)


    with open('./analysis/'+str(file_csv)+'.csv', 'w', newline = '') as f:  
        writer = csv.writer(f)
        writer.writerow(['area', 'number of times'])
        if list_test !=[]:
            for k, v in count_text_list.items():
                writer.writerow([k, v]) 

def text_print(list_test):
    T0 = time.localtime()
    dir_csv_path = './analysis'
    if not os.path.isdir(dir_csv_path):
        os.mkdir(dir_csv_path)
    print('CSV saving...')
    file_csv = time.strftime("%Y%m%d_%H%M%S", T0)
    count_text_list = {}
    if list_test !=[]:
        for i in list_test:
            if list_test.count(i) >= 1:
                count_text_list[i] = list_test.count(i)


    with open('./analysis/'+str(file_csv)+'.csv', 'w', newline = '') as f:  
        writer = csv.writer(f)
        writer.writerow(['diameter', 'number of times'])
        if list_test !=[]:
            for k, v in count_text_list.items():
                writer.writerow([k, v])

    print('CSV save done.')


def txt_file(binary_type,threshold1,threshold2,kernal_size1,kernal_size2,canny_size1,canny_size2,blur_size):

    toAdd = [binary_type,threshold1,threshold2,kernal_size1,kernal_size2,canny_size1,canny_size2,blur_size]
    # toAdd = np.array(toAdd)
    print(toAdd)
    np.save(num_filename,toAdd)
    # print("done")

def load_txt_file():
    if os.path.isfile(num_filename):
        reader0 = np.load(num_filename)
    else:
        reader0 = [0,0,0,0,0,0,0,0]

    return reader0

# ------------------------------------------影像處理UI------------------------------------

def callback():
    pass



def main_function(img):  #主程式，所有按鍵指令都需在此UI下運行
    Num_set = load_txt_file()
    print(img.shape[1])
    
    # if img.shape[1]<=960:
    #     img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)))
    # elif img.shape[1]>=960:
    #     img = cv2.resize(img, (int(img.shape[1]*0.3), int(img.shape[0]*0.3)))
    # ---------------------------------------------------------
    # --圖像預處理，設置其大小
    # height, width = img.shape[:2]
    # size = (int(width * 0.3), int(height * 0.3))
    # img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    # ------------------------------------------------------------
    # if __name__ == '__main__':
    # ROI = img.copy()
    # cv2.namedWindow('src')
    # cv2.setMouseCallback('src', on_mouse)
    # cv2.imshow('src', img)
    # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # (x, y, w, h)  = cv2.selectROI("ROI_image", img, False, False)
    # while (x, y, w, h) == (0, 0, 0, 0):
    #     (x, y, w, h)  = cv2.selectROI("ROI_image", img, False, False)
    #     img0 = img[y : y+h, x:x+w]
    #     img0 = cv2.resize(img, (int(img.shape[1]*1), int(img.shape[0]*1)))
    #     print(img.shape)

    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #灰階處理
        knn_img = knn(gray_img)

    cv2.namedWindow('setting')
    cv2.resizeWindow('setting',500,400)
    cv2.createTrackbar('binary', 'setting', 0, 4, callback)
    cv2.createTrackbar('binary1', 'setting', 0, 255, callback)
    cv2.createTrackbar('binary2', 'setting', 0, 255, callback)
    cv2.createTrackbar('erode', 'setting', 0, 50, callback)
    cv2.createTrackbar('dilate', 'setting', 0, 50, callback)
    cv2.createTrackbar('Canny1', 'setting', 0, 255, callback)
    cv2.createTrackbar('Canny2', 'setting', 0, 255, callback)
    cv2.createTrackbar('blur', 'setting', 0, 150, callback)


    cv2.setTrackbarPos('binary', 'setting', Num_set[0])
    cv2.setTrackbarPos('binary1', 'setting', Num_set[1])
    cv2.setTrackbarPos('binary2', 'setting', Num_set[2])
    cv2.setTrackbarPos('erode', 'setting', Num_set[3])
    cv2.setTrackbarPos('dilate', 'setting', Num_set[4])
    cv2.setTrackbarPos('Canny1', 'setting', Num_set[5])
    cv2.setTrackbarPos('Canny2', 'setting', Num_set[6])
    cv2.setTrackbarPos('blur', 'setting', Num_set[7])

    while (True):
        binary_type = cv2.getTrackbarPos('binary', 'setting')  
        threshold1 = cv2.getTrackbarPos('binary1', 'setting')
        threshold2 = cv2.getTrackbarPos('binary2', 'setting')
        kernal_size1 = cv2.getTrackbarPos('erode', 'setting')
        kernal_size2 = cv2.getTrackbarPos('dilate', 'setting')
        canny_size1 = cv2.getTrackbarPos('Canny1', 'setting')
        canny_size2 = cv2.getTrackbarPos('Canny2', 'setting')
        blur_size0 = cv2.getTrackbarPos('blur', 'setting')
        blur_size = blur_size0*2+1
        bin_img = binarization(knn_img,threshold1,threshold2,binary_type)
        dilate_img = dilate(bin_img,kernal_size2)
        # erode_img = erode(dilate_img,kernal_size1)
        canny_image = erode(dilate_img,kernal_size1)
        # canny_image = cv2.Canny(erode_img, canny_size1, canny_size2)
        # cv2.imshow('img0',canny_image)
        if blur_size != 0:
            canny_image = blur(canny_image,blur_size)
        # cv2.imshow('canny_image',canny_image)
        contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        con = cv2.drawContours(img.copy(), contours, -1, (0,0,255), 1)

        if len(canny_image.shape) == 2:
            canny_image0 = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR)
        if con.shape[0] > con.shape[1]:
            imgconbine = np.hstack((con, canny_image0))
        else:
            imgconbine = np.vstack((con, canny_image0))
        #cv2.imshow('image', con)
        #cv2.imshow('img1',canny_image)


        # cv2.waitKey(0)
        # cv2.destroyWindow('img0')
        cv2.imshow('imgconbine_img',imgconbine)
    
        # if cv2.waitKey(1) == ord("r"):

        #     img = cv2.imread('S__425246902.jpg')
        #     # ---------------------------------------------------------
        #     # --圖像預處理，設置其大小
        #     # height, width = img.shape[:2]
        #     # size = (int(width * 0.3), int(height * 0.3))
        #     # img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        #     # ------------------------------------------------------------
        #     ROI = img.copy()
        #     cv2.namedWindow('src')
        #     cv2.setMouseCallback('src', on_mouse)
        #     cv2.imshow('src', img)
        #     cv2.waitKey(0)
        #     # cv2.destroyAllWindows()

        if cv2.waitKey(1) == ord("w"):  #顯示各段影像處理成果
            cv2.imshow('img01',gray_img)
            cv2.imwrite('./pic/'+fileName2+'_gray.jpg', gray_img)
            cv2.waitKey(0)
            cv2.destroyWindow('img01')
            cv2.imshow('img02',bin_img)
            cv2.imwrite('./pic/'+fileName2+'_bin.jpg', bin_img)
            cv2.waitKey(0)
            cv2.destroyWindow('img02')
            # cv2.imshow('img03',erode_img)
            # cv2.imwrite('./pic/'+fileName2+'_erode.jpg', erode_img)
            # cv2.waitKey(0)
            # cv2.destroyWindow('img03')
            cv2.imshow('img04',dilate_img)
            cv2.imwrite('./pic/'+fileName2+'_dilate.jpg', dilate_img)
            cv2.waitKey(0)
            cv2.destroyWindow('img04')

        elif cv2.waitKey(1) == ord('t'):  #紀錄當下UI參數
            dir_list_path = './num'
            if not os.path.isdir(dir_list_path):
                os.mkdir(dir_list_path)
            print('Data saving......')
            print(binary_type,threshold1,threshold2,kernal_size1,kernal_size2,canny_size1,canny_size2,blur_size0)
            txt_file(binary_type,threshold1,threshold2,kernal_size1,kernal_size2,canny_size1,canny_size2,blur_size0)
            print('......Done.')
            # window.destroy
            
        elif cv2.waitKey(1) == ord('q'):  #關閉當下處理中的圖片
            print("quit")
            cv2.destroyAllWindows()
            # window.destroy
            break
            
        # elif cv2.waitKey(1) == ord('s'):
        #     # _,contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #     # con = cv2.drawContours(img, contours, -1, (255,154,152), 3)
        #     # cv2.imshow('01',con
        # ) # 打開預覽視窗
        #     for cnt in contours:
        #         area = cv2.contourArea(cnt)
        #         print('Area size:'+ str(area))
        #         contours_list.append(area)
        #     max_area_list = sorted(contours_list, reverse = True)
        #     area_list2 = max_area_list[2:]
        #     print(area_list2)
        #     sum_area = sum(area_list2)
        #     print(sum_area)
        #     print(max_area_list[0])
        #     percentage = sum_area/max_area_list[0]*100
        #     print('Area percentage: {:.2f}'.format(percentage)+'%')

        elif cv2.waitKey(1) == ord('a'):  #繪製空洞面積
            T = time.localtime()
            print('start to draw the area.....')
            for cnt in contours:
                area = cv2.contourArea(cnt)
                #print('pixel size:',area)
# -------------------------單位轉換(Pixel to Minimeter)------------------------------------------
                if area >= 1:
                 # print(area)
                    #high = 18  #單位:cm
                    #scope = (-3.39*high+78)
                    scope = 13.11875
                    area = area/scope 
# --------------------------------------------------------------------------------修改
                #print('Area size:'+ str(area))
                contours_list.append(area)
            #print(contours_list)
            if contours != []:
                img_area = img.copy()
                for i, c in enumerate(contours):
                    # draw_circle2 = draw_area(con,img0,c,contours_list[i])
                    # cv2.imshow('test',draw_circle2)
                    # print(draw_circle2.shape)
                    # print(contours_list[i])

                    if int(contours_list[i]) >= 400:
                        # print(contours_list[i])
                        area_color = (0,0,255)
                    elif int(contours_list[i]) >= 10:
                        area_color = (0,255,255)
                    elif int(contours_list[i]) >= 1:
                        area_color = (0,255,0)
                    else:
                        area_color = (255,0,0)
                    # print(area_color)
                    # print(contours) 
                    area_img = cv2.drawContours(img_area, contours[i], -1, area_color, 2)                

            cv2.putText(area_img, '>= 400 mm^2', (10,20) , cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.putText(area_img, '>= 10 mm^2', (10,40) , cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
            cv2.putText(area_img, '>= 1 mm^2', (10,60) , cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            cv2.putText(area_img, '< 1 mm^2', (10,80) , cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
            # max_area_list = sorted(contours_list, reverse = True)
            text_print1(contours_list)
            print('1..2..3...draw the area done.')
            fileName2 = time.strftime("%Y%m%d_%H%M%S", T)
            # cv2.imwrite('D:/NYCU/Pythonwork/catch_percent/'+fileName2+'_area.jpg', area_img)
            cv2.imwrite('./pic/'+fileName2+'_area.jpg', area_img)
            cv2.imshow('draw_circle_area',area_img)
            # area_list2 = max_area_list[2:]
            # print(area_list2)
            # sum_area = sum(area_list2)
            # print(sum_area)
            # print(max_area_list[0])
            # percentage = sum_area/max_area_list[0]*100
            # print('Area percentage: {:.2f}'.format(percentage)+'%')
                    # cv2.imshow('image', con)
            # cv2.imwrite('./pic/'+fileName2+'canny.jpg',canny_image)
            # cv2.imwrite('./pic/'+fileName2+'binarization.jpg',bin_img)
            # cv2.imwrite('./pic/'+fileName2+'erode.jpg',erode_img)
            # cv2.imwrite('./pic/'+fileName2+'dilate.jpg',dilate_img)
            # cv2.imwrite('./pic/'+fileName2+'blur.jpg',canny_image)

        elif cv2.waitKey(1) == ord('d'):  #繪製空洞直徑(內接圓)
            T = time.localtime()
            print('start to draw the diameter.....')
            dir_pic_path = './pic'
            if not os.path.isdir(dir_pic_path):
                os.mkdir(dir_pic_path)

            if contours != []:
                for i, c in enumerate(contours):
                    draw_circle = draw(con,gray_img,c)
            elif contours == []:
                draw_circle = con
            text_print(text_list)
            print('1..2..3...draw the diameter done.')
            fileName = time.strftime("%Y%m%d_%H%M%S", T)
            cv2.imwrite('./pic/'+fileName+'.jpg', draw_circle)
            cv2.imshow('draw_circle_diameter',draw_circle)
def select_file():
    file_path = tkinter.filedialog.askopenfilename(title = "Select picture file",filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("bmp file","*.bmp"),("all files","*.*")))
    print(file_path)
    if file_path is '':
        return
    # return file_path
    img = cv2.imread(file_path)
    main_function(img)
    

# 制作按键

button = tk.Button(window,text = '選擇圖片',font =('Arial',12),width = 10,height = 1,command = select_file)
button.pack(side = LEFT,padx = 50)
button2 = tk.Button(window,text = '拍攝圖片',font =('Arial',12),width = 10,height = 1,command = ROI_capture)
button2.pack(side = RIGHT,padx = 50)


window.mainloop()












