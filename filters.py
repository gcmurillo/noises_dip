
import noises
import cv2
import numpy as np
from matplotlib import pyplot as plt

def maximun_filter(image, mask_size=3):
    temp_image = np.float64(np.copy(image))
    new_image = np.float64(np.copy(image))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    # print(temp_image[0:10, 0:10])
    # print(temp_image[0:3, 0:4])
    # print(np.max(temp_image[0:3, 0:4]))

    if len(temp_image.shape) == 2:  # si es de un solo canal
        for i in range(h - 1):
            for j in range(w - 1):
                mask = temp_image[i:i+mask_size, j:j+mask_size]
                maximum = np.max(mask)
                new_image[i+(mask_size//2), j+(mask_size//2)] = maximum
    # else:
    #     for i in range(h - 1):
    #         for j in range(w - 1):
    #             mask_0 = temp_image[i:i+mask_size, j:j+mask_size,0]
    #             maximum_0 = np.max(mask_0)
    #             mask_1 = temp_image[i:i+mask_size, j:j+mask_size,1]
    #             maximum_1 = np.max(mask_1)
    #             mask_2 = temp_image[i:i+mask_size, j:j+mask_size,2]
    #             maximum_2 = np.max(mask_2)
    #             new_image[i+(mask_size//2), j+(mask_size//2), 0] = maximum_0
    #             new_image[i+(mask_size//2), j+(mask_size//2), 1] = maximum_1
    #             new_image[i+(mask_size//2), j+(mask_size//2), 2] = maximum_2

    return new_image


def minimun_filter(image, mask_size=3):
    temp_image = np.float64(np.copy(image))
    new_image = np.float64(np.copy(image))

    h = temp_image.shape[0]
    w = temp_image.shape[1]

    if len(temp_image.shape) == 2:  # si es de un solo canal
        for i in range(h - 1):
            for j in range(w - 1):
                mask = temp_image[i:i+mask_size, j:j+mask_size]
                minimum = np.min(mask)
                new_image[i+(mask_size//2), j+(mask_size//2)] = minimum

    return new_image


def punto_medio_filter(image, mask_size=3):
    temp_image = np.float64(np.copy(image))
    new_image = np.float64(np.copy(image))

    h = temp_image.shape[0]
    w = temp_image.shape[1]

    if len(temp_image.shape) == 2:  # si es de un solo canal
        for i in range(h - 1):
            for j in range(w - 1):
                mask = temp_image[i:i+mask_size, j:j+mask_size]
                minimum = np.min(mask)
                maximum = np.max(mask)
                medio = (minimum + maximum) / 2
                new_image[i+(mask_size//2), j+(mask_size//2)] = medio

    return new_image




image = cv2.imread('image1.jpg',0)
noisy = noises.gaussian_noise(image)
restaured = punto_medio_filter(noisy)
restaured = punto_medio_filter(restaured)
plt.subplot(1,3,1),plt.imshow(image, cmap = 'gray')
plt.title('Source'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(noisy, cmap = 'gray')
plt.title('Noisy'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(restaured, cmap = 'gray')
plt.title('Restaured'), plt.xticks([]), plt.yticks([])
plt.show()
