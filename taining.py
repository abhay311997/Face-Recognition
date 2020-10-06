# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:42:27 2020

@author: abhay
"""
import numpy as np
import cv2
import os

import  faceRecognition as fr

test_img = cv2.imread(r'C:\Users\abhay\Desktop\FaceRecog\testimage.jpg')
 
faces_detected,gray_img = fr.faceDetection(test_img) 

#print("Face Detected: ",faces_detected)
#training will begin from here

faces,faceID = fr.labels_for_training_data(r'C:\Users\abhay\Desktop\FaceRecog\images')
face_recognizer = fr.train_Classifier(faces, faceID)
face_recognizer.save(r'C:\Users\abhay\Desktop\FaceRecog\trainingData.yml')

name={0:'Abhay',1:'Tom cruise'}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+w,x:x+h]
    label,confidence = face_recognizer.predict(roi_gray)
    print("label: ",label)
    print("confidence: ",confidence)
    fr.draw_rect(test_img, face)
    predict_name = name[label]
    fr.put_text(test_img,predict_name, x, y)
    
resized_img = cv2.resize(test_img,(700,700))

cv2.imshow('image',resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

    