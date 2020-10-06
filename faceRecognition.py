"""
Created on Sun May 24 12:08:07 2020

@author: abhay
"""
#lbph method-- LOCAL BINARY PIXEL HISTOGRAM
#min pixal value 0 max 255, threshold between 0-255
#threshold is set: to take important feature nd ignore less import feature
import numpy as np        #to in array matrix
import cv2          #to convert pic in pixel form
import os

def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    #face_haar = cv2.CascadeClassifier(r'C:\Users\abhay\Desktop\FaceRecog\haarcascade_frontalface_alt.xml')
    face_haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_haar.detectMultiScale(gray_img,scaleFactor =1.3,minNeighbors=3)
    return faces,gray_img


def labels_for_training_data(directory):
    faces = []
    faceID = []
    
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping System File")
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path,filename)
            print("img_path",img_path)
            print("id",id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print("Not Loaded Properly")
                continue
            
            faces_rect,gray_img = faceDetection(test_img)
            (x,y,w,h) = faces_rect[0]
            roi_gray = gray_img[y:y+w,x:x+h] #roi- region of intersest
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID


def train_Classifier(faces,faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h) = face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=3)
    
def put_text(test_img,label_name,x,y):
    cv2.putText(test_img,label_name,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),3)

                                