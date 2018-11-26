import numpy as np
import cv2
lst = []
aux = []
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('data/mouth.xml')
img = cv2.imread('resources/friends2.jpg')
height, width = img.shape[:2]
x = int(width / 2)
y = int(height / 2)
img = cv2.resize(img,(0,0),fx=0.75,fy=0.75)
img = cv2.bilateralFilter (img, 5, 75, 75)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3,5)
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    fx,fy=roi_gray.shape[:2]
    
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3,5)
    for (ex, ey, ew, eh) in eyes:
        if(ey<(fy/2)):
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), -1)
    
    lst.clear()
    aux.clear()
    mouth = mouth_cascade.detectMultiScale(roi_gray, 1.3,5)
    
    for(mx, my, mw, mh) in mouth:
        lst.append([mx, my, mx + mw, my + mh])
        aux.append(my + mh)
        max_h = np.max(aux)

    for i in lst:
        if(i[3]==max_h and i[1]>(fy/2)):
            cv2.rectangle(roi_color, (i[0],i[1]), (i[2],i[3]), (0, 0, 255), -1)
            
    
cv2.imshow('procesada', img)
cv2.waitKey(0)
cv2.destroyAllWindows()