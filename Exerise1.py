# Import functions and libraries
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import dct, idct
from numpy import r_
from skimage.io import imread
from skimage.color import rgb2gray


def dct_apply(image):
    size = image.shape
    dct_image = np.zeros(size)
    block = 8

    # loop through all the 8x8 blocks of the original image
    # apply the dct to the said block
    for i in r_[:size[0]:block]:
        for j in r_[:size[1]:block]:
            dct_image[i:(i+block), j:(j+block)] = dct(dct(image[i:(i+block), j:(j+block)], axis=0, norm='ortho'), axis=1, norm='ortho')
    return dct_image


def idct_apply(image):
    size = image.shape
    idct_img = np.zeros(size)
    block = 8

    thresh = 0.012
    dct_thresh = image * (abs(image) > (thresh * np.max(image)))

    for i in r_[:size[0]:block]:
        for j in r_[:size[1]:block]:
            idct_img[i:(i + block), j:(j + block)] = idct(idct(dct_thresh[i:(i+block), j:(j+block)], axis=0, norm='ortho'), axis=1, norm='ortho')

    return idct_img


def watermark(a):
    counter = 1
    maxi = 11

    for i in range(1-a.shape[0], a.shape[0]):
        diag = np.diagonal(a[::-1, :], i)[::(2 * (i % 2) - 1)]
        for j in diag:
            if counter < maxi + 1:
                index = np.where(a == j)
                a[index] = j * (1 + 0.02 * (np.random.normal(0, 1, 1)))
            else:
                index = np.where(a == j)
                a[index] = 0
            counter += 1
    return a


image = rgb2gray(imread("Hugo_Simberg-The_Wounded_Angel.jpg").astype(float))

dct_image = dct_apply(image)

# Display the original image in greyscale
plt.figure(figsize=(15, 10), constrained_layout=False)
plt.subplot(111), plt.imshow(image, "gray"), plt.title("Original greyscale image")
plt.show()

# Display DCT image
plt.figure()
plt.imshow(dct_image, cmap='gray')
plt.title("2D DCT of the image")
plt.show()

size = image.shape
for i in r_[:size[0]:8]:
    for j in r_[:size[1]:8]:
        watermark(dct_image[i:(i + 8), j:(j + 8)])

idct_image = idct_apply(dct_image)

difference_image = image - idct_image

# Display DCT image
plt.figure()
plt.imshow(dct_image, cmap='gray')
plt.title("2D DCT of the image, with watermark")
plt.show()

# Display IDCT image
plt.figure()
plt.imshow(idct_image, cmap='gray')
plt.title("2D IDCT of the image, with watermark")
plt.show()

# Display Difference image
plt.figure()
plt.imshow(difference_image, cmap='gray')
plt.title("Difference image")
plt.show()

histogram, bin_edges = np.histogram(difference_image, bins=256, range=(0, 1))
plt.figure()
plt.plot(histogram)
plt.title("Histogram of difference image")
plt.show()

"""
--------------------------PART 2--------------------------
"""

