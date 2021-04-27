import cv2
import numpy as np
import pickle
import os
#implementing recognizer
recognizer=cv2.face.LBPHFaceRecognizer_create()
#loading the trained files
recognizer.read("trainner.yml")
labels={}
with open("labels.pickle", 'rb') as f:
    #this opos_labebs is in a form of {“name”: id}, but we need to reverse it
    opos_labels=pickle.load(f)
    #reversing the labels form
    labels={v:k for k,v in opos_labels.items()}
   
face_cascade=cv2.CascadeClassifier('/home/pi/Desktop/cam/camera/new_haarcascade_frontalface_default.xml')

cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

while True:
    ret, img = cam.read()
    img=cv2.flip(img,-1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (fx,fy,fw,fh) in face:
        cv2.rectangle(img,(fx,fy),(fx+fw,fy+fh),(0,255,0),2)
        #converting the image into gray
        fac_gr=gray[fy:fy+fh,fx:fx+fw]
        #recognizer.predict return the confidence level that how much is the match
        id_, conf=recognizer.predict(fac_gr)
        if conf>=90:      #if confidence is over 90% then put the label as a text on the top of the frame
            cv2.putText(img, labels[id_], (fx,fy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "unkown", (fx,fy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow('img',img)
    if cv2.waitKey(20) & 0xFF==ord('s'):
        break

cam.release()
cv2.destroyAllWindows()