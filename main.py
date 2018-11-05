import cv2
import numpy as np
aux = []
lst = []
face_cascade = cv2 . CascadeClassifier ('data/haarcascade_frontalface_default.xml') 
eye_cascade = cv2 . CascadeClassifier ('data/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('data/mouth.xml') 
capture = cv2.VideoCapture(0)
cv2.namedWindow("Video:", cv2.WINDOW_AUTOSIZE)

while(True):
    
    ret, frame = capture.read()
    height, width = frame.shape[:2]
    x_f = int(width / 1.5)  # reduce el size del frame
    y_f = int(height / 1.5)  # reduce el size del frame
    frame = cv2.resize(frame, (x_f, y_f), interpolation=cv2.INTER_AREA)
    # frame = cv2.subtract(frame,50) aumenta el brillo
    frame = cv2.bilateralFilter(frame, 5, 75, 75)  # se aplica un filtro al frame para reducir el ruido
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]  # se crea un roi_cara en escala de grises
        roi_color = frame[y:y + h, x:x + w]
        fx, fy = roi_gray.shape[:2]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)  # se encarga de detectar los ojos dentro del roi_cara, devuelve una lista de rectangulos
        
        for (ex, ey, ew, eh) in eyes:  # recorremos cada rectangulo 
            
            if(ey < (fy / 2)):  # valida de tal forma que solo muestra aquellos rectangulos que estan por arriba de la mitad de la cara (ojos)
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), -1)  # dibujamos el rectangulo en las ubicaciones correspondientes
        lst.clear()
        aux.clear()    
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for(mx, my, mw, mh) in mouth:
            lst.append([mx, my, mx + mw, my + mh])  # se crea una lista con coordenadas de los vertices de rectangulos
            aux.append(my + mh)  # se crea una lista con todas las altura de los rectangulos
            max_h = np.max(aux)  # con la ayuda de numpy se determina cual es el rectangulo mas cercano al borde inferior
            
# valida de tal forma que solo muestra aquellos rectangulos que estan por debajo de la mitad de la cara (boca) y  esten mas cercanos al borde inferior del roi_cara   
        for i in lst:
            if(i[3] == max_h and i[1] >= (fy / 2)): 
                cv2.rectangle(roi_color, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), -1)
        
    cv2.imshow("Video:", frame)
    key = cv2.waitKey(33)
    if (key == 27):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
capture.release()

