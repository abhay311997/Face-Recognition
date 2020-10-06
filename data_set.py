"""
Created on Sun May 24 12:24:21 2020

@author: abhay
"""

import  cv2
import sys

cpt = 0
vidstream = cv2.VideoCapture(0)
while True:
    ret,frame = vidstream.read() #read frame and return code
    cv2.imshow("Test frame",frame) #show image in window
    
    cv2.imwrite(r"C:\Users\abhay\Desktop\FaceRecog\images\0\image%04i.jpg" %cpt,frame)
    cpt+=1
    
    if cv2.waitKey(10)==ord('q'):
        cv2.destroyAllWindows()
        vidstream.release()
        break
    