import cv2
import numpy as np
from PIL import Image, ImageFilter
aux = []
lst = []
face_cascade = cv2 . CascadeClassifier ('data/haarcascade_frontalface_default.xml') 
eye_cascade = cv2 . CascadeClassifier ('data/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('data/mouth.xml') 

print("0: Ninguno, 1: Gaussiano, 2: Sal y Pimienta" )
opcion = input("Seleccione el tipo de ruido: ")

capture = cv2.VideoCapture("resources/video.mp4")
width = capture.get(3)
height = capture.get(4)
cv2.namedWindow("Video:", cv2.WINDOW_AUTOSIZE)

while(True):
    
    ret, frame = capture.read()
        
    if(frame is not None):
        frame = cv2.resize(frame,(0,0),fx=0.70,fy=0.70)

        # Ruido Gauss
        if opcion == "1":
            row, col, ch = frame.shape
            mean = 0
            sigma = 0.9
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            gauss = np.uint8(gauss)
            frame = cv2.add(frame, gauss)
            cv2.imshow("Ruido Gauss:", frame)

        # Ruido Sal y Pimienta
        if opcion == "2":
            row, col, ch = frame.shape

            s_vs_p = 0.5
            amount = 0.08
            noisy = np.copy(frame)
            # Sal
            num_salt = np.ceil(amount * frame.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in frame.shape]
            noisy[tuple(coords)] = 255

            # Pimienta
            num_pepper = np.ceil(amount * frame.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in frame.shape]
            noisy[tuple(coords)] = 0
            frame = noisy

            cv2.imshow("Ruido Sal y Pimienta:", frame)

        # Filtro Mediano
        if opcion != "0":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            frame = frame.filter(ImageFilter.MedianFilter(5))
            frame = np.array(frame)
            # Convert RGB to BGR
            frame = frame[:, :, ::-1].copy()
            

        frame = cv2.subtract(frame,10)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    
        faces = face_cascade.detectMultiScale(gray,1.3,5)
    
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0),0)
            roi_gray = gray[y:y + h, x:x + w]  # se crea un roi_cara en escala de grises
            roi_color = frame[y:y + h, x:x + w]
            fx, fy = roi_gray.shape[:2]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3,5)  # se encarga de detectar los ojos dentro del roi_cara, devuelve una lista de rectangulos
        
            for (ex, ey, ew, eh) in eyes:  # recorremos cada rectangulo 
                if(ey < (fy / 2)):  # valida de tal forma que solo muestra aquellos rectangulos que estan por arriba de la mitad de la cara (ojos)
                    
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), -1) #dibujamos el rectangulo en las ubicaciones correspondientes
                                                    #
                    
            lst.clear()
            aux.clear()    
            mouth = mouth_cascade.detectMultiScale(roi_gray, 1.3,5)
            for(mx, my, mw, mh) in mouth:
                lst.append([mx, my, mx + mw, my + mh])  # se crea una lista con coordenadas de los vertices de rectangulos
                aux.append(my + mh)  # se crea una lista con todas las alturas de los rectangulos
                max_h = np.max(aux)  # con la ayuda de numpy se determina cual es el rectangulo mas cercano al borde inferior
                
    # valida de tal forma que solo muestra aquellos rectangulos que estan por debajo de la mitad de la cara (boca) y  esten mas cercanos al borde inferior del roi_cara   
            for i in lst:
                if(i[3] == max_h and i[1] >= (fy / 2)): 
                    cv2.rectangle(roi_color, (i[0], i[1]+3), (i[2], i[3]-15), (0, 0, 255), -1)
         
        
        cv2.imshow("Video:",frame)
        key = cv2.waitKey(10)
        if (key == 27):
            break
        
    else:
        break

cv2.imshow("cara oculta",roi_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
capture.release()




