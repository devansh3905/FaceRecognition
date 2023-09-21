import numpy as np
import cv2
import os

face_Cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
face_id = input('\n enter user id - ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

cap.set(3,640) # set Width
cap.set(4,480) # set Height
count = 0
while True:
    #capture frame by frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_Cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces:
        color = (255,0,0)
        stroke = 2
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        encode_cord_x = x+w
        encode_cord_y = y+h
        cv2.rectangle(frame, (x, y), (encode_cord_x, encode_cord_y), color, stroke)
        count += 1  # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])



        #display resulting frame
        cv2.imshow('frame',frame)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 20:  # Take 30 face sample and stop video
        break


cap.release()
cv2.destroyAllWindows()