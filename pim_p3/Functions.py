import numpy as np
import interactive_only as interactive


class Node:
    """
        Class node to store information
    """
    def __init__(self, value, i, j):
        """
        Node that save information like the pixel value, position i-j matrix and the next node
        next to him.
        :param value: pixel value
        :param i: position i matrix
        :param j: position j matrix
        """
        self.value = value
        self.i = i
        self.j = j
        self.next = None


def get_mask(image):
    """
    Get the polygon referent to the piece of image that we want to delete. And we convert it into
    a mask.
    :param image: Image where we want to delete something
    :return: mask with the piece of image that we want to delete
    """
    # Getting the area to delete from the image
    rdi = interactive.get_mouse_click(image)
    polygon = rdi.points
    # Getting the deleting mask
    return interactive.compute_mask(image.shape[1], image.shape[0], polygon)


def compute_gradient_image(gray_image):
    """
    Here we compute the gradient approximation of an image from horizontal and vertical derivative approximations
    G = sqrt(Gx**2 + Gy**2)
    :param gray_image: image on gray scale
    :return: gradient magnitude of each pixel
    """
    dx, dy = np.gradient(gray_image)
    return np.sqrt(np.power(dx, 2) + np.power(dy, 2))


def compute_minimum_energy_matrix(matrix):
    """
    Here we compute the minimum energy matrix from the image gradient
    :param matrix: the gradient image
    :return: minimum energy matrix
    """
    n_rows = len(matrix)
    n_columns = len(matrix[0])
    for i in range(1, n_rows):
        for j in range(0, n_columns):
            try:
                matrix[i, j] = matrix[i, j] + np.min([matrix[i - 1, j - 1], matrix[i - 1, j], matrix[i - 1, j + 1]])
            except IndexError:
                matrix[i, j] = matrix[i, j] + np.min([matrix[i - 1, j - 1], matrix[i - 1, j]])
    return matrix


def min_path(matrix):
    """
    Here we look for a path from bottom to up pixel by the minimum value neighbour of it.
    :param matrix: the minimum energy matrix
    :return: the first node to continue the path by the next function node
    """
    i = len(matrix) - 1
    row = matrix[i]
    j = np.where(row == np.min(row))[0][0]
    min_value = np.min(row)

    node = Node(min_value, i, j)
    first_node = node
    node_previus = node

    for i in range(len(matrix) - 2, -1, -1):
        if j - 1 == -1:
            min_value = min((matrix[i, j], j),
                            (matrix[i, j + 1], j + 1))
        elif j + 1 == len(matrix[0]):
            min_value = min((matrix[i, j - 1], j - 1),
                            (matrix[i, j], j))
        else:
            min_value = min((matrix[i, j - 1], j - 1),
                            (matrix[i, j], j),
                            (matrix[i, j + 1], j + 1))

        j = min_value[1]

        node = Node(min_value[0], i, j)
        node_previus.next = node
        node_previus = node

    return first_node


def get_minpath_mem(image, with_mask=False, mask=None):
    """
    From a image this function return the optimal seam
    :param image: image in
    :param with_mask: if we want to compute the image with a mask
    :param mask: wich mask it is
    :return: the first node seam path
    """
    gradient_img = compute_gradient_image(np.mean(image, 2)/255.)
    if with_mask:
        gradient_img[mask == 1] = -100

    return min_path(compute_minimum_energy_matrix(gradient_img))