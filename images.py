import cv2
import os
import numpy as np
from PIL import Image
import pickle

#the path where this .py file is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#looking for a foldere named 'images' from the path
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/cam/camera/new_haarcascade_frontalface_default.xml')

#OpenCV recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0  #id for labeling
label_ids = {}   #empty label dictionary
labels_y = []   #this is used to append ids
train_x = []   #this is used to append faces
#walking through each file in the image folder
for root, dirs, files in os.walk(image_dir):
    #going through each file in files
    for file in files:
        #deals with only the files with .jpg or .png extensions
        if file.endswith("png") or file.endswith("jpg"):
            #getting the path of file
            path = os.path.join(root, file)
            #labeling it with the folder name
            label = os.path.basename(os.path.dirname(path))
            #if label does not have an id in the library, assign a current id and increment the current_id 
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            #taking the id (new one or previously existing) from the label_ids dictionary   
            idf = label_ids[label]
            #taking the image from path and convert("L") is for gray scale
            pil_image = Image.open(path).convert("L") 
            #resizing the image to get better performance
            #these dimentions are taken according to the experimental results
            size = (1000, 1000)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            #training image into a numpy array
            image_array = np.array(final_image, "uint8")
            #finding region of interest (faces)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.3, minNeighbors=5)

            #going through faces one by one
            for (x,y,w,h) in faces:
                fc = image_array[y:y+h, x:x+w] #region of interest
                #we append the face and its id to the train_x and labels_y
                train_x.append(fc)
                labels_y.append(idf)

#saving the labels into a .pickle file
with open("labels.pickle", 'wb') as f:
    #add the label_ids dictionary to the file caller labels,pickle
    pickle.dump(label_ids, f)
#training the recognizer with faces and ids
recognizer.train(train_x, np.array(labels_y))
#savin the trained recognizer into a trainer.yml file
recognizer.save("trainer.yml")

print("Finished")
