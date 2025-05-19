import cv2 as cv 
import os
import numpy as np 

names = ["nayan" , "rash","sam" , "vijay"]
dir = r"facerecognition\train"
haarcascade = cv.CascadeClassifier("haarcasecade.xml")
features = [] 
labels = [] 

def create_train() :
    for person in names : 
        path = os.path.join(dir , person) 
        label = names.index(person)
        for img_path in os.listdir(path):
            image = cv.imread(os.path.join(path,img_path))
            gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)
            face_rect = haarcascade.detectMultiScale(gray , 1.1 , 4)
            for (x,y,w,h) in face_rect : 
                face = gray[y:y+h , x:x+w]
                features.append(face)
                labels.append(label)
            
create_train()

features = np.array(features , dtype='object')
labels = np.array(labels)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)

face_recognizer.save('facesTrained.yml')
np.save('features.npy' , features)
np.save('labels.npy' , labels)
