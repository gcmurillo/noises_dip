
import noises
import cv2
import numpy as np
from matplotlib import pyplot as plt

def maximun_filter(image, mask_size=3):
    temp_image = np.float64(np.copy(image))
    new_image = np.float64(np.copy(image))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    print(temp_image[0:10, 0:10])
    print(temp_image[0:3, 0:4])
    print(np.max(temp_image[0:3, 0:4]))

    if len(temp_image.shape) == 2:  # si es de un solo canal
        for i in range(h - 1):
            for j in range(w - 1):
                mask = temp_image[i:i+mask_size, j:j+mask_size]
                maximum = np.max(mask)
                new_image[i+(mask_size//2), j+(mask_size//2)] = maximum

    return new_image


image = cv2.imread('image1.jpg',0)
noisy = noises.pepper_noise(image)
restaured = maximun_filter(noisy)
plt.subplot(1,3,1),plt.imshow(image, cmap = 'gray')
plt.title('Source'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(noisy, cmap = 'gray')
plt.title('Noisy'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(restaured, cmap = 'gray')
plt.title('Restaured'), plt.xticks([]), plt.yticks([])
plt.show()
