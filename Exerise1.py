# Import functions and libraries
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import dct, idct
from numpy import r_
from skimage.io import imread
from skimage.color import rgb2gray


image = rgb2gray(imread("Hugo_Simberg-The_Wounded_Angel.jpg").astype(float))


def dct2(a):
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a):
    return idct(idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


size = image.shape
dct_image = np.zeros(size)
block = 8

# loop through all the 8x8 blocks of the original image
# apply the dct to the said block
for i in r_[:size[0]:block]:
    for j in r_[:size[1]:block]:
        dct_image[i:(i+block), j:(j+block)] = dct2(image[i:(i+block), j:(j+block)])

# Display the original image in greyscale
plt.figure(figsize=(15, 10), constrained_layout=False)
plt.subplot(111), plt.imshow(image, "gray"), plt.title("Original greyscale image")
plt.show()

# Display DCT image
plt.figure()
plt.imshow(dct_image, cmap='gray', vmax=np.max(dct_image)*0.01, vmin=0)
plt.title("2D DCT of the image, with 8x8 blocks")
plt.show()
