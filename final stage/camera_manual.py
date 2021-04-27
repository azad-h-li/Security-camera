import cv2 
import os 
import numpy as np 
import pickle
import datetime
import playsound #library to play sound
import threading #used torunbackground functions
import time 
import pygame #for playing sound
pygame.init()

#implementing recognizer
recognizer=cv2.face.LBPHFaceRecognizer_create()
#loading the trained files
recognizer.read("trainner.yml")
labels={}
with open("labels.pickle", 'rb') as f:
    #this opos_labebs is in a form of {“name”: id}, but we need to reverse it
    og_labels=pickle.load(f)
    #reversing the label form
    labels={v:k for k,v in og_labels.items()}
    
#loading the cascade file   
face_cascade=cv2.CascadeClassifier('/home/pi/Desktop/cam/camera/new_haarcascade_frontalface_default.xml')

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

#10 seconds counter for fire alarm
def timer():
    global t
    t=10
    while t>0:
        time.sleep(1)
        t=t-1       


#loading yolo trained files
def load_yolo():
    net2 = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net2.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net2.getUnconnectedOutLayers()]
    return net2, classes, output_layers

#detecting guns and fire
def detect_objects(img, net2, outputLayers):            
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net2.setInput(blob)
    outputs = net2.forward(outputLayers)
    return blob, outputs

#operations on the image and security systems
def operations_on_image(outputs, height, width, classes, img):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)

                
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            alarm_flag=0
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x,y), (x+w, y+h), (200,50,100), 2)
            cv2.putText(img, label, (x, y - 5), font, 1, (200,50,100), 2)
            if label=="Fire":      #incase fire detected
                print("Fire")      #Print “Fire” on the terminal
                alarm=pygame.mixer.Sound("AlarmSound.mp3")       #load the alarm sound
                if t==0:         #bool_flag show if alarm is on or not. output of counter
                    alarm.play()       #fire alarm on      
                    count=threading.Thread(target=timer)     #introducing times as a thread
                    count.start()       #after 10 seconds t value will be 0 again


            else:    #incase gun is detected
                print("Gun detected")

    return img      

#loading the model and data
model, classes, output_layers = load_yolo()

t=0    #gloval variable used for counter

cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
print("\nCamera is operating now!\n")

while True:
    ret, img = cam.read()
    img=cv2.flip(img,-1)                          #flipping the image, due to the position of camera
    cv2.rectangle(img,(210,150),(430,330),(100,100,0),1)      #imaginary rectangle for object tracking
    
    #face and identity recognition on the image
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.5, 5)
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

    #pedestrian detection            
    #the list of class names that need to be detected
    objects = ['person']
    counter=0    #to count number of people on the image
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
                counter = counter + 1
    
    #Fire and Gun detection
    #getting image data
    height, width, channels = img.shape
    #detecting objects
    blob, outputs = detect_objects(img, model, output_layers)
    #drawing box and doing the secutiy operations
    img = operations_on_image(outputs, height, width, classes, img)

    #displayin date, time and crowdedness informations on the window
    dt = str(datetime.datetime.now())
    cv2.putText(img, dt, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 10, 100), 2, cv2.LINE_AA)
    people = str(counter) + " people"
    cv2.putText(img, people, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 10, 10), 1, cv2.LINE_AA)    
    cv2.imshow('img',img)

    if cv2.waitKey(20) & 0xFF==ord('s'):
        break
cam.release()
cv2.destroyAllWindows()
