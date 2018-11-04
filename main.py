import cv2
import numpy as np
aux=[]
list=[]
face_cascade = cv2 . CascadeClassifier ('data/haarcascade_frontalface_default.xml') 
eye_cascade = cv2 . CascadeClassifier ('data/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('data/mouth.xml') 
capture = cv2.VideoCapture(0)
cv2.namedWindow("Video:", cv2.WINDOW_AUTOSIZE)

while(True):
    
    ret, frame = capture.read()
    height, width = frame.shape[:2]
    x = int(width /1.5)
    y = int(height /1.5)
    frame = cv2.resize(frame, (x, y), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if(not ret):
        break
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), -1)
            mouth = mouth_cascade.detectMultiScale(roi_gray, 1.3)
    
            for(mx, my, mw, mh) in mouth:

                list.append([mx, my, mx + mw, my + mh])
                aux.append(my + mh)
                max_h = np.max(aux)
                for i in list:
                    if(i[3] == max_h):
                        cv2.rectangle(roi_color, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), -1)
        
    cv2.imshow("Video:", frame)
    key = cv2.waitKey(33)
    if (key == 27):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
capture.release()

