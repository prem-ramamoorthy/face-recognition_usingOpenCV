import cv2 as cv
import numpy as np 

people = ["nayan" , "rash","sam" , "vijay"]

face_detector = cv.CascadeClassifier("haarcasecade.xml")
face_recognition = cv.face.LBPHFaceRecognizer_create()
face_recognition.read(r"facerecognition\facesTrained.yml")

#Vijay
vijay_img = cv.imread(r"facerecognition\test\vijay-test\images (7).jpg")
vijay_gray = cv.cvtColor(vijay_img,cv.COLOR_BGR2GRAY)
vijay_face_rect = face_detector.detectMultiScale(vijay_gray , 1.1 , 4)

for (x,y,w,h) in vijay_face_rect : 
    vijay_face = vijay_gray[y:y+h , x:x+w]
    label , confidence = face_recognition.predict(vijay_face)
    text = str(people[label]) + "Confidence : " + str(confidence)
    print(text)
    cv.putText(vijay_img , text , (20,20), cv.FONT_HERSHEY_COMPLEX_SMALL , 0.70 , (0,0,0) , 1)
    cv.rectangle(vijay_img , (x,y) , (x+w,y+h) , (0,255,0) , 2)

cv.imshow("Detected Face " , vijay_img)
cv.waitKey(0)
