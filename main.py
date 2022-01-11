import cv2
import mediapipe as mp
from past.builtins import raw_input
import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture("pro.mp4")
detector = HandDetector(detectionCon=0.8, maxHands=2)




while True:
    success, img = cap.read()


    hands, img = detector.findHands(img)  # With Draw
    # hands = detector.findHands(img, draw=False)  # No Draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmarks points
        bbox1 = hand1["bbox"]  # Bounding Box info x,y,w,h

        # crop_img = img[y:y + h, x:x + w]
        # cv2.imshow("cropped", crop_img)
        centerPoint1 = hand1["center"]  # center of the hand cx,cy
        handType1 = hand1["type"]  # Hand Type Left or Right

        # print(len(lmList1),lmList1)
        print(bbox1)
        # print(centerPoint1)
        fingers1 = detector.fingersUp(hand1)
        # length, info, img = detector.findDistance(lmList1&#91;8], lmList1&#91;12], img) # with draw
        # length, info = detector.findDistance(lmList1&#91;8], lmList1&#91;12])  # no draw

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmarks points
            bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h
            centerPoint2 = hand2["center"]  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type Left or Right

            fingers2 = detector.fingersUp(hand2)
            # print(fingers1, fingers2)
            # length, info, img = detector.findDistance(lmList1&#91;8], lmList2&#91;8], img) # with draw
            length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)  # with draw

    cv2.imshow("Image", img)
    cv2.waitKey(500)

# class handDetector():
#     def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
#         self.mode = mode
#         self.maxHands = maxHands
#         self.detectionCon = detectionCon
#         self.trackCon = trackCon
#
#         self.mpHands = mp.solutions.hands
#         self.hands = self.mpHands.Hands(self.mode, self.maxHands,
#                                         self.detectionCon, self.trackCon)
#         self.mpDraw = mp.solutions.drawing_utils
#
#     def findHands(self, img, draw=True):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.hands.process(imgRGB)
#         # print(results.multi_hand_landmarks)
#
#         if self.results.multi_hand_landmarks:
#             for handLms in self.results.multi_hand_landmarks:
#                 if draw:
#                     self.mpDraw.draw_landmarks(img, handLms,
#                                                self.mpHands.HAND_CONNECTIONS)
#         return img
#
#     def findPosition(self, img, handNo=0, draw=True):
#
#         lmList = []
#         if self.results.multi_hand_landmarks:
#             myHand = self.results.multi_hand_landmarks[handNo]
#             for id, lm in enumerate(myHand.landmark):
#                 # print(id, lm)
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 # print(id, cx, cy)
#                 lmList.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
#
#         return lmList
fn = "tayninh.jpg"
# img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

# cv1.imshow("anh", img)
# cv1.waitKey()
# img_binary = cv1.adaptiveThreshold(img, maxValue=120, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=15, C=8)
# cv1.imshow("Binary Image", img_binary)
# cv1.waitKey()
# fn_new = "n.jpg"
# cv1.imwrite(fn_new, img_binary)



# cap = cv2.imread(fn)
# imgRGB = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(static_image_mode=False,
#                       max_num_hands=2,
#                       min_detection_confidence=0.5,
#                       min_tracking_confidence=0.5)
# mpDraw = mp.solutions.drawing_utils
# imgRGB = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
# results = hands.process(imgRGB)
# if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             for id, lm in enumerate(handLms.landmark):
#                 #print(id,lm)
#
#                 h, w, c = cap.shape
#                 cx, cy = int(lm.x *w), int(lm.y*h)
#                 #if id ==0:
#                 cv2.circle(cap, (cx,cy), 3, (255,0,255), cv2.FILLED)
#
#
#
#             mpDraw.draw_landmarks(cap, handLms, mpHands.HAND_CONNECTIONS)
#
# dim = (1000, 1000)
# # cv2.rectangle(cap,cx, cy,(255, 0, 0),3)
# resized = cv2.resize(cap, dim, interpolation = cv2.INTER_AREA)
# cv2.imshow("Test", resized)
#
# cv2.waitKey(0)
