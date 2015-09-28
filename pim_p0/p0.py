
# coding: utf-8

## Ejercicio 1: creación de vectores y matrices

#### 1.1 Crea un vector fila de 10 elementos con valores enteros aleatorios (random) entre 0 y 100 y realiza las siguientes operaciones sobre el vector creado:

# In[105]:

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
get_ipython().magic(u'matplotlib inline')


# In[ ]:

vector_fila = np.random.randint(0, 100, 10)
print vector_fila


# In[ ]:

#a) accede a la tercera posición; 
vector_fila[2]


# In[ ]:

#b) obtén el vector formado por las 5 primeras posiciones del vector original; 
vector_fila[:5]


# In[ ]:

#c) obtén el vector formado por las posiciones impares del vector original; 
vector_fila[::2]


# In[ ]:

#d) suma todos los elementos del vector.
vector_fila.sum()


#### 1.2 Crea 2 matrices ‘a’ y ‘b’, la primera de dos filas y 3 columnas (2x3): [1 2 3; 4 5 6] y la segunda de (3x2): [2 4; 5 7; 9 3], a continuación realiza las siguientes operaciones:

# In[ ]:

a = np.matrix('1 2 3; 4 5 6')
b = np.matrix('2 4; 5 7; 9 3')


# In[ ]:

#a) multiplica las dos matrices
np.dot(a,b)


# In[ ]:

#b) convierte la matriz ‘a’ en una nueva matriz ‘a2’ (3x2) utilizando la función RESHAPE
#(utiliza la función help en la Command Window para utilizar RESHAPE correctamente);
#qué diferencia hay entre la matriz a2 y la traspuesta de a? 

a2 = a.reshape((3,2))
a3 = a.transpose()
print a
print a2
print a3


# In[ ]:

#c  suma las matrices ‘a2’ y ‘b’ y guarda el resultado en la variable ‘c’;

c = a2 + b


# In[ ]:

#d multiplica, elemento por elemento, las matrices ‘a2’ y ‘b’.

np.multiply(a2, b)


## Ejercicio 2: creación de imágenes

#### Crea una función edgeImg() que implemente los siguientes puntos

# In[ ]:

#a) crear una imagen de tamaño 256x256 px de un solo canal de tipo 8-bit
# unsigned mitad blanca y mitad negra, tal como se muestra en Figura 1;

edge_img = np.zeros((256, 256), np.uint8)
edge_img[:256,:128] = 1


# In[ ]:

#b) visualizar la imagen y guardarla como edge_img.jpg.
plt.imshow(edge_img, cmap='gist_gray')
plt.imsave("edge_img.jpg", edge_img, cmap='gist_gray')


## Ejercicio 3: tratamiento de imágenes en escala de grises 

#### Dada la imagen en escala de grises hand_1C.jpg, crear la función rotateImg() que implemente los siguientes puntos: 

# In[65]:

#a) abrir el archivo y guardarlo en una variable; 
hand_1C = plt.imread('hand_1C.jpg')


# In[22]:

#b) visualizar la imagen
plt.imshow(hand_1C, cmap='gist_gray')


# In[40]:

#c) visualizar en una gráfica de tipo ‘plot’ los valores de
#los niveles de gris de la fila 200 en el rango de columnas [310, 330];

v = hand_1C[200, 310:330]
plt.plot(hand_1C[200, 310:330])


# In[33]:

#d) convertir la imagen original de manera que resulte especular
#respecto al eje vertical;
hand_1C_especular = hand_1C[::,::-1]
plt.imshow(hand_1C_especular, cmap='gist_gray')


# In[44]:

#e) rotar la imagen original de 90º en dirección
#contraria a las agujas del reloj; 
hand_1C_vertical = hand_1C.T[::-1, ::]
plt.imshow(hand_1C_vertical, cmap='gist_gray')


## Ejercicio 4: binarización de imágenes

# La binarización BI(x,y) de una imagen I(x,y) a partir de un umbral TH consiste en convertir la imagen en una imagen binaria (de 0s y 1s), dependiendo de si el nivel de intensidad de cada pixel de la imagen original es mayor o menor al umbral Th.
# Dada la imagen hand_1C.jpg, crear la función B = thresholdImg(I, th) que aplica el umbral th a la imagen I para crear su versión binaria B (como en Figura 3). 

# In[77]:

def thresholdImg(I, th):
    b = np.copy(I)
    b[b < th] = 0
    b[b > th] = 255
    return b


# In[78]:

binary = thresholdImg(hand_1C, 128)
plt.imshow(binary, cmap='gist_gray')


## Ejercicio 5: creación de imágenes de 3 canales (en color)

# Las imágenes RGB están formadas por 3 matrices, llamadas comúnmente canales. Como ejercicio práctico para familiarizar con las imágenes de 3 canales, implementar los siguientes puntos: 

# In[95]:

#a) crea las imágenes en escala de grises mostradas 
#en Figura 4 (a-c) (tamaño 128x128 px); 
a = np.zeros((128,128), 'uint8')
b = np.zeros((128,128), 'uint8')
c = np.zeros((128,128), 'uint8')

a[::, 64::] = 255
b[64::, ::] = 255
c[:64:, :64:] = 255


# In[101]:

#b) combina las 3 imágenes de forma que se obtenga la
#imagen mostrada en la Figura 4d

d = np.zeros((128,128,3), 'uint8')
d[::, :: , 0] = a
d[::, :: , 1] = b
d[::, :: , 2] = c
plt.imshow(d)


## Ejercicio 6: tratamiento de imágenes en color RGB

# Dados los dos archivos de imagen hand.jpg (Figura 5 a) y mapfre.jpg (Figura 5 b), crear la función fuseImg() que implemente los siguientes puntos: 

# In[102]:

#a) abrir los dos archivos 
hand = plt.imread('hand.jpg')
mapfre = plt.imread('mapfre.jpg')


# In[124]:

#b) convertir la imagen hand.jpg en escala de grises
#utilizando la función rgb2gray;
hand_gray = rgb2gray(hand)


# In[125]:

#c) realizar la binarización sobre la imagen resultante
#en b) para conseguir 2 regiones: una perteneciente a la
#mano (foregorund) y la otra al fondo (background);
binary_hand = thresholdImg(hand_gray, 0.5)


# In[126]:

plt.imshow(binary_hand, cmap='gist_gray', vmin = 0, vmax= 1 )


# In[153]:

#d) utilizar la matriz binaria creada en c) para
#fusionar las imágenes hand y Mapfre (Fig. 5 c)

d = np.zeros(mapfre.shape)


# In[148]:

mapfre.shape


# In[149]:

a = binary_hand == 0


# In[151]:

a.shape


# In[ ]:



