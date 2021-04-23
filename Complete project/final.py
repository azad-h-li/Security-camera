import cv2
import os
import numpy as np
import RPi.GPIO as GPIO
import pickle
import datetime
import playsound
import threading
import time
import pygame
pygame.init()

#servo motors control
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11,GPIO.OUT)
servo_up_down=GPIO.PWM(11,50)
GPIO.setup(12,GPIO.OUT)
servo_left_right=GPIO.PWM(12,50)
servo_up_down.start(0)
time.sleep(0.2)
servo_left_right.start(0)
time.sleep(0.2)
DC=7
servo_up_down.ChangeDutyCycle(DC)
time.sleep(0.02)
servo_up_down.ChangeDutyCycle(0)
flag1servo=0

#identity recognition
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels={}
with open("labels.pickle", 'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
    
#face recognition using Haar feature-based cascade file   
face_cascade=cv2.CascadeClassifier('/home/pi/Desktop/cam/camera/new_haarcascade_frontalface_default.xml')

#pedestrian detection using TensorFlow
classNames= []
classFile= 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net1 = cv2.dnn_DetectionModel(weightsPath, configPath)
net1.setInputSize(320, 320)
net1.setInputScale(1.0 / 127.5)
net1.setInputMean((127.5, 127.5, 127.5))
net1.setInputSwapRB(True)

#Fire and Gun detection and security systems
def timer():
    global t
    t=10
    while t>0:
        time.sleep(1)
        t=t-1
    bool_flag=0

def load_yolo():
    net2 = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net2.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net2.getUnconnectedOutLayers()]
    return net2, classes, output_layers

def detect_objects(img, net2, outputLayers):            
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net2.setInput(blob)
    outputs = net2.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width, classes, img, bool_flag, DC):
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
            if label=="Gun":
                dist_horizontal=320-(x+w/2)
                dist_vertical=240-(y+h/2)
                time.sleep(0.02)
                if dist_horizontal>110:
                    servo_left_right.ChangeDutyCycle(7.3)
                    time.sleep(0.005)
                    servo_left_right.ChangeDutyCycle(0)
                if dist_horizontal<-110:
                    servo_left_right.ChangeDutyCycle(6.7)
                    time.sleep(0.005)
                    servo_left_right.ChangeDutyCycle(0)
                if (dist_vertical>90 and DC<12):
                    DC=DC+0.1
                    servo_up_down.ChangeDutyCycle(DC)
                    time.sleep(0.02)
                    servo_up_down.ChangeDutyCycle(0)
                if (dist_vertical<-90 and DC>2):
                    DC=DC-0.1
                    servo_up_down.ChangeDutyCycle(DC)
                    time.sleep(0.02)
                    servo_up_down.ChangeDutyCycle(0)

            if label=="Fire":
                alarm_flag=1
                print("Fire")
                alarm=pygame.mixer.Sound("AlarmSound.mp3")
                if bool_flag==0 and alarm_flag==1:
                    alarm.play()
                    bool_flag=1
                    count=threading.Thread(target=timer)
                    count.start()
                    print(bool_flag)
                    print(t)
                if alarm_flag==0 or t==0:
                    alarm.stop()
                    bool_flag=0
                        
    return img, bool_flag, DC

model, classes, output_layers = load_yolo()
global alarm_flag
bool_flag=0



print("\nCamera is operating now!\n")

cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

while True:
    ret, img = cam.read()
    img=cv2.flip(img,-1)
    cv2.rectangle(img,(210,150),(430,330),(100,100,0),1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #face and identity recognition on the image
    face = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (fx,fy,fw,fh) in face:
        cv2.rectangle(img,(fx,fy),(fx+fw,fy+fh),(0,255,0),2)
        fac_gr=gray[fy:fy+fh,fx:fx+fw]
        fc=img[fy:fy+fh,fx:fx+fw]
        id_, conf=recognizer.predict(fac_gr)
        if conf>=70:
            cv2.putText(img, labels[id_], (fx,fy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            
    #pedestrian detection on the image
    objects = ['person']
    counter = 0
    classIds, confs, bbox = net1.detect(img, 0.45, 0.2)
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, box in zip(classIds.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                counter = counter + 1
    
    #Fire and Gun detection
    height, width, channels = img.shape
    blob, outputs = detect_objects(img, model, output_layers)
    img, bool_flag, DC = get_box_dimensions(outputs, height, width, classes, img, bool_flag, DC)

    #displayin basic informations on the screen
    dt = str(datetime.datetime.now())
    cv2.putText(img, dt, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 10, 100), 2, cv2.LINE_AA)
    
    people = str(counter) + " people"
    cv2.putText(img, people, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 10, 10), 1, cv2.LINE_AA)
    
    cv2.imshow('img',img)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
