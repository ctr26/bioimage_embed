import numpy as np


def create_circle_contour(radius, image_size):
    image = np.zeros(image_size, dtype=np.uint8)
    center = (image_size[0] // 2, image_size[1] // 2)

    y, x = np.ogrid[: image_size[0], : image_size[1]]
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    image[np.abs(distance - radius) < 1] = 255  # Set contour to white

    return image, distance
