import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy import signal as sg


class Node:
    def __init__(self, value, i, j):
        self.value = value
        self.i = i
        self.j = j

# Loading image
image = np.array(Image.open('countryside.jpg'))

# Image to grayscale
gray_image = np.mean(image, 2)/255.

# Horizontal derivative approximations
sobel_x = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]

# Vertical derivative approximations
sobel_y = [[1, 2, 1],
           [0, 0, 0],
           [-1, -2, -1]]

# Calculating the gradient image
gradient_image = np.sqrt(np.power(sg.convolve2d(gray_image, sobel_x, "same"), 2)
                         + np.power(sg.convolve2d(gray_image, sobel_y, "same"), 2))

# Calculating the M minimum energy matrix
minimum_energy_m = np.zeros(gradient_image.shape)
minimum_energy_m[0, :] = gradient_image[0, :]

for i in range(1, len(minimum_energy_m)):
    for j in range(0, len(minimum_energy_m[0])):
        try:
            minimum_energy_m[i, j] = gradient_image[i, j] + np.min([minimum_energy_m[i - 1, j - 1],
                                                                    minimum_energy_m[i - 1, j],
                                                                    minimum_energy_m[i - 1, j + 1]])
        except IndexError:
            minimum_energy_m[i, j] = gradient_image[i, j] + np.min([minimum_energy_m[i - 1, j - 1],
                                                                    minimum_energy_m[i - 1, j]])

# Doing backtracking to find the minimum path from bottom to top
no_blocking = False

if no_blocking:
    plt.figure(1)
    plt.show(block=False)

cont = 1
while cont <= 90:
    min_path = []

    i = len(minimum_energy_m) - 1
    row = minimum_energy_m[i]
    j = np.where(row == np.min(row))[0][0]
    min_value = np.min(row)

    min_path.append(Node(min_value, i, j))

    for i in range(len(minimum_energy_m) - 2, -1, -1):
        min_value = min((minimum_energy_m[i, j - 1], j - 1),
                        (minimum_energy_m[i, j], j),
                        (minimum_energy_m[i, j + 1], j + 1))
        j = min_value[1]

        min_path.append(Node(min_value[0], i, j))

    _image = np.zeros((image.shape[0], image.shape[1] - 1, 3)).astype('uint8')
    _minimum_energy_m = np.zeros((minimum_energy_m.shape[0], minimum_energy_m.shape[1] - 1))

    i = 0
    for row in range(image.shape[0] - 1, -1, -1):
        j = min_path[i].j

        _image[row, :, 0] = np.delete(image[row, :, 0], j)
        _image[row, :, 1] = np.delete(image[row, :, 1], j)
        _image[row, :, 2] = np.delete(image[row, :, 2], j)

        _minimum_energy_m[row] = np.delete(minimum_energy_m[row], j)

        image[row, j, :] = 0.

        i += 1
    if no_blocking:
        plt.imshow(image)
        plt.draw()

    image = _image[:, :, :]
    minimum_energy_m = _minimum_energy_m[:, :]
    cont += 1

# Showing results
#plt.figure(1)
#plt.imshow(gradient_image, cmap='gray')
#plt.figure(2)
#plt.imshow(minimum_energy_m, cmap='gray')
#plt.figure(3)
#plt.imshow(image)
#plt.show()


