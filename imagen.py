import numpy as np
import cv2
from scipy import fftpack
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt

lst = []
aux = []
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('data/mouth.xml')
profile_cascade = cv2.CascadeClassifier('data/profile.xml')
img = cv2.imread('resources/friends2.jpg')
height, width = img.shape[:2]
x = int(width / 2)
y = int(height / 2)
img = cv2.resize(img,(0,0),fx=0.75,fy=0.75)

print("0: Ninguno, 1: Gaussiano, 2: Sal y Pimienta")

def aplicar_filtro(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    img = img.filter(ImageFilter.MedianFilter(5))
    img = np.array(img)
    # Convert RGB to BGR
    img = img[:, :, ::-1].copy()
    
    return img


def detectar(img,titulo):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(0,2):
        faces = face_cascade.detectMultiScale(gray, 1.2,5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0),1)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            fx,fy=roi_gray.shape[:2]
            
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.2,5)
            for (ex, ey, ew, eh) in eyes:
                if(ey<(fy/2)):
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), -1)
            
            lst.clear()
            aux.clear()
            mouth = mouth_cascade.detectMultiScale(roi_gray, 1.2,5)
            
            for(mx, my, mw, mh) in mouth:
                lst.append([mx, my, mx + mw, my + mh])
                aux.append(my + mh)
                max_h = np.max(aux)
        
            for i in lst:
                if(i[3]==max_h and i[1]>(fy/2)):
                    cv2.rectangle(roi_color, (i[0],i[1]), (i[2],i[3]), (0, 0, 255), -1)
        img=cv2.flip(img,0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          
    cv2.imshow(titulo,img)
    cv2.waitKey(0)
    
    
opcion = input("Seleccione el tipo de ruido: ")

if opcion == "1":
    row, col, ch = img.shape
    mean = 0
    sigma = 0.8
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    gauss = np.uint8(gauss)
    img = cv2.add(img, gauss)
    img2= cv2.add(img,gauss)
    detectar(img,"Ruido Gaussiano")
    detectar(aplicar_filtro(img2),"Sin Ruido Gaussiano")
    
    
if opcion == "0":
    detectar(aplicar_filtro(img),"Sin Ruido")

# Ruido Sal y Pimienta
if opcion == "2":
    row, col, ch = img.shape
    s_vs_p = 0.7
    amount = 0.1
    noisy = np.copy(img)
    # Sal
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in img.shape]
    noisy[tuple(coords)] = 255

    # Pimienta
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in img.shape]
    noisy[tuple(coords)] = 0
    img = noisy
    img2= noisy
    detectar(img,"Ruido Sal-Pimienta")
    detectar(aplicar_filtro(img2),"Sin Ruido Sal-Pimienta")





if opcion =="3":
    
    image=np.mean(cv2.imread('resources/friends2.jpg'),axis=2)/255
    f1=fftpack.fft2(image.astype(np.float32))
    f2=fftpack.fftshift(f1)
    magnitud = (20*np.log10(0.1+f2)).astype(int)
    plt.rcParams["figure.figsize"] = (15,8)
    plt.subplot(221),plt.imshow(magnitud, cmap = 'gray')
    plt.title('Espectro Inicial'), plt.xticks([]), plt.yticks([])
    for n in range(image.shape[1]):
        image[:,n]= image[:,n]+(np.cos(1.5*np.pi*n))
    
    plt.subplot(222),plt.imshow(image, cmap = 'gray')
    plt.title('Ruido Periodico'), plt.xticks([]), plt.yticks([])
    f1=fftpack.fft2(image.astype(np.float32))
    f2=fftpack.fftshift(f1)
    magnitud = (20*np.log10(0.1+f2)).astype(int)
    plt.subplot(223),plt.imshow(magnitud, cmap = 'gray')
    plt.title('Espectro con Ruido'), plt.xticks([]), plt.yticks([])
    f2[270:340,:570]=f2[270:340,630:]=f2[:260,590:613,]=f2[340:,590:613]=0
    image2=fftpack.ifft2(fftpack.ifftshift(f2)).real
    plt.subplot(224),plt.imshow(image2,cmap = 'gray')
    plt.title('Resultado Final'), plt.xticks([]), plt.yticks([])
    plt.show()
    i1=cv2.imread("resources/ruido.png")
    i2=cv2.imread("resources/sin_ruido.png")
    detectar(i1,"Ruido Periodico")
    detectar(i2,"Sin Ruido Periodico")
    
    
    
 
    
cv2.destroyAllWindows()



  


