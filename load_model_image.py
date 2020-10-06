# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:54:16 2020

@author: abhay
"""

import numpy as np
import cv2
import os

import  faceRecognition as fr
print (fr)

test_img = cv2.imread(r'C:\Users\abhay\Desktop\FaceRecog\testimage.jpg')

 
faces_detected,gray_img = fr.faceDetection(test_img)

print("Face Detected: ",faces_detected)

face_recognizer= cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\abhay\Desktop\FaceRecog\trainingData.yml')

name={0:'Abhay',1:'xyz'}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+w,x:x+h]
    label,confidence = face_recognizer.predict(roi_gray)
    print("label: ",label)
    print("confidence: ",confidence)
    fr.draw_rect(test_img, face)
    predict_name = name[label]
    #less confidence more better
    if(confidence>75):
        fr.put_text(test_img, 'Unknown', x, y)
        continue
    fr.put_text(test_img,predict_name, x, y)
    
resized_img = cv2.resize(test_img,(700,700))

cv2.imshow('image',resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

    