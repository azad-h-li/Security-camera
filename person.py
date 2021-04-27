import cv2
import os
import numpy as np

#class names from coco.names will be copied to classNames
classNames= []
classFile= 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#introduction mobilenet and graph files
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

#creating model
net1 = cv2.dnn_DetectionModel(weightsPath, configPath)
#these configurations below are from a documentation
net1.setInputSize(320, 320)
net1.setInputScale(1.0 / 127.5)
net1.setInputMean((127.5, 127.5, 127.5))
net1.setInputSwapRB(True)

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

while True:
    ret, img = cam.read()
    img=cv2.flip(img,-1) 

    #the list of class names that need to be detected
    objects = ['person']
    #getting the prediction form the image
    classIds, confs, bbox = net1.detect(img, 0.45, 0.2)
    if len(classIds) != 0:
        #taking one id from the classids (clasids are lattened to be used)
        for classId, box in zip(classIds.flatten(), bbox):
            #classId starts from one, but array starts from 0, that’s why 1 is substracted from classId
            className = classNames[classId - 1]
            #if the detected object is desired object
            if className in objects:
                #draw the rectangle box on the image
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(20) & 0xFF == ord('s'):
        break

cam.release()
cv2.destroyAllWindows()
