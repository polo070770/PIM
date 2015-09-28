import math
import numpy as np
from scipy import signal as sg

#GAUSSIAN FUNCTION
def gaussian(x, y, size, sigma):
    return (1. / (2. * math.pi * sigma * sigma)) * math.exp(- (math.pow(x-(size/2.),2) + math.pow(y-(size/2.),2)) / (2. * sigma * sigma))

#GAUSSIAN FILTER FUNCTION
def gaussian_filter(sigma):
    size = int(6 * sigma + 1)
    gauss_matrix = np.zeros( (size, size) )
    
    for x in range(size):
        for y in range(size):
            gauss_matrix[x][y] = gaussian(x, y, size, sigma)
    return gauss_matrix / sum(sum(gauss_matrix)) #Normalizado

#CONVOLUTION ONE CHANNEL FUNCTION
def convolution2d(img, kernel):
    matrix = sg.convolve2d(img, kernel, "same")
    return matrix
    
def smoothing(img, sigma):
    kernel = gaussian_filter(sigma)
    return convolution2d(img, kernel)
    
def downsampling(img):
    h = img.shape[0]    
    w = img.shape[1]
    
    if(h % 2 != 0): h -= 1
    if(w % 2 != 0): w -= 1
    
    m = np.zeros((h/2, w/2))    
    
    _i = 0
    for i in range(0,h, 2):
        _j = 0
        for j in range(0, w, 2):
            m[_i, _j] = img[i, j]
            _j += 1
        _i += 1
        
    return m
    
def normalizeIt(matrix):
    """
        Function that normalize a matrix and then return it.
    """
    matrix -= np.min(matrix)
    matrix /= np.max(matrix)
    matrix -= np.max(matrix)
    return matrix
    
def cutIt(matrix, x, y, width, height):
    """
        Function that returns a IN matrix copy crop it with the IN values
    """
    return normalizeIt(np.copy(matrix[x:x + height, y:y + width]))
            
def cross_correlate(search_img, template_img):
    """
        Function that returns the correlation of two matrix.
    """
    return sg.correlate2d(search_img.astype('float'), template_img.astype('float'), mode='full', boundary='fill', fillvalue=0)
    
def pyramidal(img, nivell):
    #x = x0 - (s_w / nivell)
    #y = y0 - (s_h / nivell)
    
    #size_width = s_w * nivell
    #size_height = s_h * nivell
    
    #matrix = cutIt(img, x, y, size_width, size_height)

    i = 0    

    o0 = 2 ** nivell    

    while (i < nivell):

        matrix = smoothing(img, o0)        
        matrix = downsampling(matrix)
        
        o0 /= 2        
        
        i += 1
        
    return matrix