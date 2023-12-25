import cv2
import mediapipe as mp  
import time

cap = cv2.VideoCapture(0)

mpMaos = mp.solutions.hands
maos = mpMaos.Hands()
mpDesenha = mp.solutions.drawing_utils

xTempo = 0
yTempo = 0

while True: #iniciando o loop principal 
    sucesso,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    resultado = maos.process(imgRGB)

    if resultado.multi_hand_landmarks:
        for maosLms in resultado.multi_hand_landmarks:
            for id , lm in enumerate(maosLms.landmark):
            #            print(id, lm)

                h, w ,  c = img.shape
                cx , cy = int(lm.x * w) , int(lm.y * h)
                print(id, cx ,cy)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        mpDesenha.draw_landmarks(img, maosLms , mpMaos.HAND_CONNECTIONS)

    yTempo = time.time()
    fps = 1 / (yTempo - xTempo)
    xTempo = yTempo

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

