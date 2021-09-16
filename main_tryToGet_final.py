from os import close
import cv2
import numpy as np
import csv
import time
import tkinter as tk
import tkinter.filedialog 
import os


window = tk.Tk()
window.title('Bone Density')
window.geometry('400x200')
label = tk.Label(window,text = '開始測試' ,font = ('Arial',12),width = 30)
label.pack()
label4 = tk.Label(window,text = '功能按鍵:' ,font = ('Arial',12),width = 30)
label4.pack()
label1 = tk.Label(window,text = '按(t) 暫存調整數值' ,font = ('Arial',12),width = 30)
label1.pack()
# label2 = tk.Label(window,text = '按(s) 顯示面積比例' ,font = ('Arial',12),width = 30)
# label2.pack()
label2 = tk.Label(window,text = '按(a) 計算框選空洞面積' ,font = ('Arial',12),width = 30)
label2.pack()
label3 = tk.Label(window,text = '按(d) 計算框選空洞大小' ,font = ('Arial',12),width = 30)
label3.pack()
label4 = tk.Label(window,text = '按(q) 關閉離開' ,font = ('Arial',12),width = 30)
label4.pack()


avg = None
xy_list = []
contours = []
contours_list = []
text_list = []
num_filename = './num/bone_density_log.npy'
Num_set = np.array(0)
# T = time.localtime()







def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype = np.uint8),-1)
    return  cv_img


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
    img = cv2.GaussianBlur(img, (kernal_size3,kernal_size3),0)
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
    cv2.putText(img, text1, maxDistPt , cv2.FONT_HERSHEY_COMPLEX, 0.5, (22,245,255), 1, cv2.LINE_AA)

    return img

def draw_area(img,img0,contours,contours_area):
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
    text1 = str(np.int(contours_area))
    text_list.append(text1)
    cv2.circle(img,maxDistPt, np.int(maxVal),(255,255,255), 1, cv2.LINE_8, 0)
    cv2.putText(img, text1, maxDistPt , cv2.FONT_HERSHEY_COMPLEX, 0.5, (22,245,255), 1, cv2.LINE_AA)

    return img

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

    # with open('./analysis/'+str(file_csv)+'.csv', 'w', newline = '') as f:  
    #     writer = csv.writer(f)
    #     writer.writerow(['diameter', 'number of times'])
    #     if list_test !=[]:
    #         for k, v in count_text_list.items():
    #             writer.writerow([k, v])

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






def callback():
    pass

def main_function():


    file_path = tkinter.filedialog.askopenfilename(title = "Select picture file",filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("bmp file","*.bmp")))
    if file_path is '':
        return
    Num_set = load_txt_file()
    img = cv_imread(file_path)
    img = cv2.resize(img, (int(img.shape[1]*0.6), int(img.shape[0]*0.6)))

    # (x, y, w, h)  = cv2.selectROI("ROI_image", img, False, False)
    # while (x, y, w, h) == (0, 0, 0, 0):
    #     (x, y, w, h)  = cv2.selectROI("ROI_image", img, False, False)
    # img0 = img[y : y+h, x:x+w]
    # img0 = cv2.resize(img, (int(img.shape[1]*1), int(img.shape[0]*1)))
    # print(img.shape)

    if len(img.shape) == 3:
        img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('setting')
    cv2.createTrackbar('binary', 'setting', 0, 4, callback)
    cv2.createTrackbar('binary1', 'setting', 0, 255, callback)
    cv2.createTrackbar('binary2', 'setting', 0, 255, callback)
    cv2.createTrackbar('erode', 'setting', 0, 50, callback)
    cv2.createTrackbar('dilate', 'setting', 0, 50, callback)
    cv2.createTrackbar('Canny1', 'setting', 0, 255, callback)
    cv2.createTrackbar('Canny2', 'setting', 0, 255, callback)
    cv2.createTrackbar('blur', 'setting', 0, 50, callback)


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

        bin_img = binarization(img0,threshold1,threshold2,binary_type)
        erode_img = erode(bin_img,kernal_size1)
        dilate_img = dilate(erode_img,kernal_size2)
        canny_image = cv2.Canny(dilate_img, canny_size1, canny_size2)
        if blur_size != 0:
            canny_image = blur(canny_image,blur_size)
        _,contours, hierarchy = cv2.findContours(canny_image, cv2.cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        con = cv2.drawContours(img.copy(), contours, -1, (255,154,152), 1)

        if len(canny_image.shape) == 2:
            canny_image0 = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR)
        if con.shape[0] > con.shape[1]:
            imgconbine = np.hstack((con, canny_image0))
        else:
            imgconbine = np.vstack((con, canny_image0))
        # cv2.imshow('image', con)
        # cv2.imshow('img0',canny_image)
        cv2.imshow('img0',imgconbine)

        if cv2.waitKey(1) == ord('t'):
            dir_list_path = './num'
            if not os.path.isdir(dir_list_path):
                os.mkdir(dir_list_path)
            print('Data saving......')
            print(binary_type,threshold1,threshold2,kernal_size1,kernal_size2,canny_size1,canny_size2,blur_size0)
            txt_file(binary_type,threshold1,threshold2,kernal_size1,kernal_size2,canny_size1,canny_size2,blur_size0)
            print('......Done.')
            # window.destroy
            
        elif cv2.waitKey(1) == ord('q'):
            print("quit")
            cv2.destroyAllWindows()
            # window.destroy
            break
            
        # elif cv2.waitKey(1) == ord('s'):
        #     # _,contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #     # con = cv2.drawContours(img, contours, -1, (255,154,152), 3)
        #     # cv2.imshow('01',con) # 打開預覽視窗
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


        elif cv2.waitKey(1) == ord('a'):
            T = time.localtime()

            print('start to draw the area.....')
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # print('Area size:'+ str(area))
                contours_list.append(area)
            if contours != []:
                for i, c in enumerate(contours):
                    draw_circle2 = draw_area(con,img0,c,contours_list[i])

            # max_area_list = sorted(contours_list, reverse = True)
            text_print(contours_list)
            print('1..2..3...draw the area done.')
            fileName2 = time.strftime("%Y%m%d_%H%M%S", T)
            cv2.imwrite('./pic/'+fileName2+'_area.jpg', draw_circle2)
            cv2.imshow('draw_circle_area',draw_circle2)
            # area_list2 = max_area_list[2:]
            # print(area_list2)
            # sum_area = sum(area_list2)
            # print(sum_area)
            # print(max_area_list[0])
            # percentage = sum_area/max_area_list[0]*100
            # print('Area percentage: {:.2f}'.format(percentage)+'%')

        elif cv2.waitKey(1) == ord('d'):
            T1 = time.localtime()

            print('start to draw the diameter.....')
            dir_pic_path = './pic'
            if not os.path.isdir(dir_pic_path):
                os.mkdir(dir_pic_path)

            if contours != []:
                for i, c in enumerate(contours):
                    draw_circle = draw(con,img0,c)
            elif contours == []:
                draw_circle = con
            text_print(text_list)
            print('1..2..3...draw the diameter done.')
            fileName = time.strftime("%Y%m%d_%H%M%S", T1)
            cv2.imwrite('./pic/'+fileName+'.jpg', draw_circle)
            cv2.imshow('draw_circle',draw_circle)

    

## 制作按键
button = tk.Button(window,text = '選擇圖片',font =('Arial',12),width = 10,height = 1,command = main_function)
button.pack()

# button0 = tk.Button(window,text = '離開',font =('Arial',12),width = 10,height = 1,command = window.destroy)
# button0.pack()

window.mainloop()












