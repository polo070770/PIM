import Functions
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time as t

# Loading image
image = np.array(Image.open('countryside.jpg'))

plt.figure(1)
plt.show(block=False)

t1 = t.clock()
cont = 1
while cont <= 40:
    current_node = Functions.get_minpath_mem(image)

    _image = np.zeros((image.shape[0], image.shape[1] + 1, 3)).astype('uint8')

    for row in range(image.shape[0] - 1, -1, -1):
        j = current_node.j

        _image[row, :, 0] = np.insert(image[row, :, 0], j+1, image[row, j, 0])
        _image[row, :, 1] = np.insert(image[row, :, 1], j+1, image[row, j, 1])
        _image[row, :, 2] = np.insert(image[row, :, 2], j+1, image[row, j, 2])

        current_node = current_node.next

        image[row, j, :] = 0.

    plt.imshow(image)
    plt.title("Iteracion %d" % cont)
    plt.draw()

    print "Iteration %d\n" % cont

    image = _image[:, :, :]
    cont += 1

t2 = t.clock()

print "Time: %.2f seconds" % (t2-t1)

# Showing results
plt.figure(1)
plt.imshow(image)
plt.show()