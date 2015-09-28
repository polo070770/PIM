import Functions
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time as t

# Loading image
image = np.array(Image.open('agbar.png'))

# Getting the deleting mask
mask = Functions.get_mask(image)

plt.figure(1)
plt.show(block=False)

t1 = t.clock()
cont = 1
while True:
    current_node = Functions.get_minpath_mem(image, True, mask)

    _image = np.zeros((image.shape[0], image.shape[1] - 1, 3)).astype('uint8')
    _mask = np.zeros((mask.shape[0], mask.shape[1] - 1))

    for row in range(image.shape[0] - 1, -1, -1):
        j = current_node.j

        _image[row, :, 0] = np.delete(image[row, :, 0], j)
        _image[row, :, 1] = np.delete(image[row, :, 1], j)
        _image[row, :, 2] = np.delete(image[row, :, 2], j)

        _mask[row] = np.delete(mask[row], j)

        image[row, j, :] = 0.

        current_node = current_node.next

    plt.imshow(image)
    plt.title("Iteration %d, remain %d points to remove\n" % (cont, len(mask[mask == 1])))
    plt.draw()

    image = _image[:, :, :]
    mask = _mask[:, :]

    print "Iteration %d, remain %d points to remove\n" % (cont, len(mask[mask == 1]))
    if len(mask[mask == 1]) == 0:
        break
    cont += 1

t2 = t.clock()

print "Time: %.2f seconds" % (t2-t1)

# Showing results
plt.figure(1)
plt.imshow(image)
plt.show()
