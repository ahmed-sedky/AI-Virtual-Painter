import cv2
import mediapipe as mp
import time
import math

class HandDetector():
    def __init__(self, mode =False ,maxHands = 2 , model_complexity = 1, detConfidence =0.5 , trackingConfidence =0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity =model_complexity
        self.detConfidence = detConfidence
        self.trackingConfidence = trackingConfidence

        self.module_hands = mp.solutions.hands
        self.object_hand = self.module_hands.Hands(self.mode , self.maxHands , self.model_complexity ,self.detConfidence ,self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipsIds = [ 4 , 8 , 12 ,16 ,20]

    def findHands(self,img ,draw =True):
        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.result = self.object_hand.process(imgRGB)
        # print(result.multi_hand_landmarks)
        if (self.result.multi_hand_landmarks):
            for handLandMark in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img , handLandMark , self.module_hands.HAND_CONNECTIONS)
        return img
    def findPosition(self,img ,hand_number =0  , draw =True):
        self.landmarks_list = []
        height ,width ,channel = img.shape
        if (self.result.multi_hand_landmarks):
            myHand = self.result.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(myHand.landmark):
                center_x, center_y = int(lm.x * width), int (lm.y *height)
                self.landmarks_list.append([id ,center_x,center_y ])
                if draw:
                    cv2.circle(img , (center_x,center_y) , 7 , (255, 0 ,255) ,  cv2.FILLED)
        return self.landmarks_list
    def fingerUp (self):
        fingers = []

        if self.landmarks_list[self.tipsIds[0]][1] > self.landmarks_list[self.tipsIds[0] - 1][1]:
                fingers.append(1)
        else:
            fingers.append(0)
        
        for id in range(1,5):
            if self.landmarks_list[self.tipsIds[id]][2] < self.landmarks_list[self.tipsIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def find_distance (self,img ,landmark1 ,landmark2 ,draw =True ,radius = 8 , line_thickness = 3 , color = (255,0,255)):
        x1 ,y1 = self.landmarks_list[landmark1][1] ,self.landmarks_list[landmark1][2]
        x2 ,y2 = self.landmarks_list[landmark2][1] ,self.landmarks_list[landmark2][2]
        cx , cy = (x1 +x2) // 2 ,(y1+y2) //2
        if draw:
            cv2.circle(img, (x1,y1) , radius , color , cv2.FILLED)
            cv2.circle(img, (x2,y2) , radius ,color , cv2.FILLED)
            cv2.line(img , (x1,y1) ,(x2,y2) ,color , line_thickness)

            cv2.circle(img, (cx,cy) , radius ,color , cv2.FILLED)
        lineLength = math.hypot(x2-x1 ,y2-y1)
        return lineLength,img , [x1,y1,x2,y2 ,cx,cy]
def main():
    cap = cv2.VideoCapture(0)
    previous_time = 0
    current_time = 0
    detector = HandDetector()
    while True:
        success ,img = cap.read()   #to show frames
        current_time = time.time()
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img)
        if (len(landmarkList) !=0 ):
            print(landmarkList[4])
        fbs = 1 / (current_time-previous_time)
        previous_time = current_time

        cv2.putText(img ,str(int(fbs)) , (10,70) ,fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX , fontScale= 2 , color= (200 , 0 ,0) , thickness= 3)
        cv2.imshow("Video" , img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()