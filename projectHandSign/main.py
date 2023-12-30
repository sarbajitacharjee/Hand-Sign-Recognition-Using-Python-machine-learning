import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np 
mp_drawing = mp.solutions.drawing_utils

def image_processed(hand_img):

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=1, min_detection_confidence=0.5)
    
    # Results
    output = hands.process(img_flip)
    # img_flip = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if output.multi_hand_landmarks:
            for num, hand in enumerate(output.multi_hand_landmarks):
                mp_drawing.draw_landmarks(hand_img, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
    hands.close()
    try:
        # data = output.multi_hand_landmarks[0]
        # #print(data)
        # data = str(data)

        # data = data.strip().split('\n')

        # garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        # without_garbage = []

        # for i in data:
        #     if i not in garbage:
        #         without_garbage.append(i)
                        
        # clean = []

        # for i in without_garbage:
        #     i = i.strip()
        #     clean.append(i[2:])

        # for i in range(0, len(clean)):
        #     clean[i] = float(clean[i])
        # return(clean)
        landmarks = output.multi_hand_landmarks[0].landmark
        keypoints = [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks]
        return keypoints , img_flip


    except:
        return(np.zeros([1,42], dtype=int)[0]) ,img_flip

import pickle
# load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)


import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
i = 0    
while True:
    
    ret, frame = cap.read()
    # frame=cv2.flip(frame,1)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # frame = cv.flip(frame,1)
    data, frame = image_processed(frame)
    
    # print(data.shape)
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1,42))
    # print(y_pred)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (50, 100)
    
    # fontScale
    fontScale = 3
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 5
    actions = ['A', 'B', 'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    # Using cv2.putText() method
    frame=cv2.flip(frame,1)
    frame = cv2.putText(frame, str(actions[y_pred[0]]), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
