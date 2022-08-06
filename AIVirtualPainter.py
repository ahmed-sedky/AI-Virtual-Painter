from email.header import Header
import numpy as np
import cv2 
import time

from sqlalchemy import false
import handTrackingModule as htm
import os

folderPath = "images/painter"
imageList = os.listdir(folderPath)
overlayImages = []
for imagePath in imageList:
    image = cv2.imread(f'{folderPath}/{imagePath}')
    resized = cv2.resize(image, (640,100), interpolation = cv2.INTER_AREA)
    overlayImages.append(resized)

cap = cv2.VideoCapture(0)
widthCam ,heightCam = 640 , 480
cap.set(3 , widthCam)
cap.set(4 , heightCam)
previous_time = 0

header = overlayImages[0]
draw_color = (0,255 ,255)
img_canvas = np.zeros((480,640,3) , np.uint8)
x_previous , y_previous = 0 , 0
detector= htm.HandDetector(detConfidence= 0.75)
while True:
    success ,img = cap.read()
    img = cv2.flip(img,1)
    img = detector.findHands(img)

    lmList = detector.findPosition(img,draw=False)
    if len(lmList)!=0 :
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        fingers = detector.fingerUp()

        if fingers[1] and fingers[2]:
            x_previous , y_previous = 0 , 0
            if y1 <100:
                if 100<x1< 200:
                    header = overlayImages[0]
                    draw_color = (0,255 ,255)
                elif 201<x1< 300:
                    header = overlayImages[1]
                    draw_color = (0,255 ,0)
                elif 301<x1< 400:
                    header = overlayImages[2]
                    draw_color = (0,0 ,255)
                elif 401<x1< 500:
                    header = overlayImages[4]
                    draw_color = (255,0 ,0)
                elif 501<x1< 600:
                    header = overlayImages[3]
                    draw_color = (0,0 ,0)
            cv2.rectangle(img,(x1,y1-25) ,(x2,y2+25) , draw_color , cv2.FILLED)
        else:
            cv2.circle(img,(x1,y1) ,10 , draw_color , cv2.FILLED )
            if x_previous ==0 and y_previous == 0:
                x_previous , y_previous = x1 ,y1
            if draw_color == (0 , 0 , 0):
                cv2.line(img,(x1,y1)  , (x_previous ,y_previous) , draw_color , 75 )
                cv2.line(img_canvas,(x1,y1)  , (x_previous ,y_previous) , draw_color , 75 )
            else:
                cv2.line(img ,  (x_previous ,y_previous),(x1,y1) , draw_color , 15 )
                cv2.line(img_canvas  , (x_previous ,y_previous),(x1,y1) , draw_color , 15 )

            x_previous , y_previous = x1 ,y1
    
    img_gray = cv2.cvtColor(img_canvas , cv2.COLOR_BGR2GRAY)
    _,img_inverse = cv2.threshold(img_gray, 50 , 255 , cv2.THRESH_BINARY_INV)
    img_inverse =cv2.cvtColor(img_inverse , cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inverse)
    img = cv2.bitwise_or(img, img_canvas)

    img[:100,:] = header
    current_time = time.time()
    fbs = 1 /(current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f'FBS: {int(fbs)}' , (400, 150) , cv2.FONT_HERSHEY_PLAIN , 3 , ( 0 ,255 ,0) , 3)
    cv2.imshow("Image" , img)
    cv2.waitKey(1)
