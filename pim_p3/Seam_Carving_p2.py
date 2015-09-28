import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy import signal as sg
import interactive_only as interactive
import time as t


class Node:
    def __init__(self, value, i, j):
        self.value = value
        self.i = i
        self.j = j
        self.next = None

# Horizontal derivative approximations
sobel_x = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]

# Vertical derivative approximations
sobel_y = [[1, 2, 1],
           [0, 0, 0],
           [-1, -2, -1]]

# Loading image
image = np.array(Image.open('agbar.png'))

# Getting the area to delete from the image
rdi = interactive.get_mouse_click(image)
polygon = rdi.points

# Getting the deleting mask
mask = interactive.compute_mask(image.shape[1], image.shape[0], polygon)

# Image to grayscale
gray_image = np.mean(image, 2) / 255.

# Calculating the gradient image
gradient_image = np.sqrt(np.power(sg.convolve2d(gray_image, sobel_x, "same"), 2) +
                         np.power(sg.convolve2d(gray_image, sobel_y, "same"), 2))

# Adjusting the values gradient mask indexes to -100
gradient_image[mask == 1] = -100.

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
    plt.figure(1, (20, 15))
    plt.show(block=False)

t1 = t.clock()

while True:
    i = len(minimum_energy_m) - 1
    row = minimum_energy_m[i]
    j = np.where(row == np.min(row))[0][0]
    min_value = np.min(row)

    node = Node(min_value, i, j)
    first_node = node
    node_previus = node

    for i in range(len(minimum_energy_m) - 2, -1, -1):
        min_value = min((minimum_energy_m[i, j - 1], j - 1),
                        (minimum_energy_m[i, j], j),
                        (minimum_energy_m[i, j + 1], j + 1))
        j = min_value[1]

        node = Node(min_value[0], i, j)
        node_previus.next = node
        node_previus = node

    _image = np.zeros((image.shape[0], image.shape[1] - 1, 3)).astype('uint8')
    _minimum_energy_m = np.zeros((minimum_energy_m.shape[0], minimum_energy_m.shape[1] - 1))
    _mask = np.zeros((mask.shape[0], mask.shape[1] - 1))

    current_node = first_node
    for row in range(image.shape[0] - 1, -1, -1):
        j = current_node.j

        _image[row, :, 0] = np.delete(image[row, :, 0], j)
        _image[row, :, 1] = np.delete(image[row, :, 1], j)
        _image[row, :, 2] = np.delete(image[row, :, 2], j)

        _minimum_energy_m[row] = np.delete(minimum_energy_m[row], j)

        _mask[row] = np.delete(mask[row], j)

        image[row, j, :] = 0.

        current_node = current_node.next

    if no_blocking:
        plt.imshow(image)
        plt.draw()

    image = _image[:, :, :]
    minimum_energy_m = _minimum_energy_m[:, :]
    mask = _mask[:, :]

    if len(mask[mask == 1]) == 0:
        break

t2 = t.clock()

print "temps:", (t2-t1)*1000

plt.figure(4)
plt.imshow(image)
plt.show()
