import cv2
import os
import numpy as np
import RPi.GPIO as GPIO
import playsound    #library to play sound
import threading     #used to run background functions
import time      
import pygame     #for playing sound
pygame.init()

#servo motors control
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11,GPIO.OUT)
servo_up_down=GPIO.PWM(11,50)
GPIO.setup(12,GPIO.OUT)
servo_left_right=GPIO.PWM(12,50)
servo_up_down.start(0)
time.sleep(0.02)
servo_left_right.start(0)
time.sleep(0.02)
DC=7
servo_up_down.ChangeDutyCycle(DC)
time.sleep(0.02)
servo_up_down.ChangeDutyCycle(0)

#10 second counter 
def timer():
    global t
    t=10
    while t>0:
        time.sleep(1)
        t=t-1
    if t==0:       #if 10 seconds over, turn the alarm off, it will start again if fire detected again
        alarm.stop()
                

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
def operations_on_image(outputs, height, width, classes, img, DC):
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
            if label=="Fire":      #in case fire detected
                print("Fire alarm on!")      #Print “Fire” on the terminal
                alarm=pygame.mixer.Sound("AlarmSound.mp3")       #load the alarm sound
                if t==0:         #bool_flag show if alarm is on or not. output of counter
                    alarm.play()       #fire alarm on      
                    count=threading.Thread(target=timer)     #introducing times as a thread
                    count.start()       #after 10 seconds t value will be 0 again


            else:    #in case gun is detected
                #measuring the distance between center coordinates of object box 
                #and imaginary rectranglular box
                dist_horizontal=320-(x+w/2)     
                dist_vertical=240-(y+h/2)
                time.sleep(0.02)
                #if the center of object box is out of the imaginary box, send a signal to the servo motors
                #to track the object
                if dist_horizontal>110:
                    servo_left_right.ChangeDutyCycle(7.8)
                    time.sleep(0.005)
                    servo_left_right.ChangeDutyCycle(7.5)
                if dist_horizontal<-110:
                    servo_left_right.ChangeDutyCycle(7.2)
                    time.sleep(0.005)
                    servo_left_right.ChangeDutyCycle(7.5)
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

    return img, DC      


#loading the model and data
model, classes, output_layers = load_yolo()

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

t=0      #global variable used for counter

while True:
    ret, img = cam.read()
    img=cv2.flip(img,-1)
    cv2.rectangle(img, (210, 150), (430, 330), (100, 100, 0), 1)   #imaginary rectangle for object tracking
    
    #getting image data
    height, width, channels = img.shape
    #detecting objects
    blob, outputs = detect_objects(img, model, output_layers)
    #drawing box and doing the secutiy operations
    img, DC = operations_on_image(outputs, height, width, classes, img, DC)
    
    cv2.imshow('img', img)

    if cv2.waitKey(20) & 0xFF == ord('s'):
        break
cam.release()
cv2.destroyAllWindows()
