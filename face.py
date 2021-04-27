import cv2
import numpy as np
import os


#loading the classifier
face_cascade=cv2.CascadeClassifier('/home/pi/Desktop/cam/camera/New folder/new_haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
while True:
    ret, img = cam.read()
    img=cv2.flip(img,-1)
    #converting the image into grayscale 
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    #detecting the faces in the image
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    #going through all the faces onebyone
    for (fx,fy,fw,fh) in face:
        #drawing a rectangle for each detected face
        cv2.rectangle(img,(fx,fy),(fx+fw,fy+fh),(0,255,0),2)
    cv2.imshow('img',img)
    if cv2.waitKey(20) & 0xFF==ord('s'):
        break
cam.release()
cv2.destroyAllWindows()
