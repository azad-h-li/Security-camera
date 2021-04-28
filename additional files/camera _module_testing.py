import cv2
import numpy as np
#accessing to the camera, in case addirional camer is used ‘0’ cav be replaced by ‘1’
cam = cv2.VideoCapture(0)
#this loop will run forever unless there is a delay proble or up press ‘s’ 
while True:
    #reading the image from camera
    ret, img = cam.read()
    #display image
    cv2.imshow('testing', img)
    #stop condition: press s button
    if cv2.waitKey(20) & 0xFF == ord('s'):
        break
#release the camera and close all the open windows
cam.release()
cv2.destroyAllWindows()
