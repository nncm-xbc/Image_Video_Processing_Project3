# Import functions and libraries
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math

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
    counter = 0
    maxi = k

    w = np.random.normal(0, 1, k)
    w_sequence = []
    for i in w:
        w_sequence.append(i)


    size = image.shape
    for x in r_[:size[0]:8]:
        for y in r_[:size[1]:8]:
            for i in range(1-a.shape[0], a.shape[0]):
                diag = np.diagonal(a[x:(x + 8), y:(y + 8)][::-1, :], i)[::(2 * (i % 2) - 1)]
                for j in diag:
                    if counter < maxi:
                        index = np.where(a == j)
                        a[index] = j * (1 + 0.02 * (w[counter]))

                    counter += 1
    return a, w_sequence


def klargest(a):
    list_coefs = []

    list_of_averages = []

    size = a.shape
    for x in r_[:size[0]:8]:
        for y in r_[:size[1]:8]:
            list_coefs_block = []
            for i in range(1 - a[x:(x + 8), y:(y + 8)].shape[0], a[x:(x + 8), y:(y + 8)].shape[0]):
                diag = np.diagonal(a[x:(x + 8), y:(y + 8)][::-1, :], i)[::(2 * (i % 2) - 1)]
                for j in diag:
                    list_coefs_block.append(j)
                list_coefs.append(list_coefs_block)

    for i in range(k):
        sum = 0.0
        for coefs_block in list_coefs:
            # print('coefs blocks: ', coefs_block)
            sum += coefs_block[i]
        average = sum/len(list_coefs)
        list_of_averages.append(average)
        print('Average value: ', average)

    return list_of_averages


def estimate_mark(averages_primary, averages_secondary):
    list_w = []
    for i in range(k):
        w = (averages_secondary[i] - averages_primary[i])/0.02 * averages_primary[i]
        list_w.append(w)
        print('Wi-k :', w)

    return list_w


def gamma(w_sequence, wi_sequence):

    w_mean = statistics.mean(w_sequence)
    wi_mean = statistics.mean(wi_sequence)

    sum_a = 0
    sum_b = 0
    sum_c = 0
    for i in range(k):
        sum_a += (wi_sequence[i] - wi_mean)*(w_sequence[i] - w_mean)
        sum_b += (wi_sequence[i] - wi_mean)**2
        sum_c += (w_sequence[i] - w_mean)**2

    val_gamma = sum_a/math.sqrt(sum_b * sum_c)

    return val_gamma


image = rgb2gray(imread("Hugo_Simberg-The_Wounded_Angel.jpg"))

k = 11

dct_image = dct_apply(image)

idct_image = idct_apply(dct_image)

dct_watermarked, w_sequence = watermark(dct_image)

idct_watermarked = idct_apply(dct_watermarked)

difference_image = image - idct_watermarked


# Display the original image in greyscale
plt.figure(figsize=(15, 10), constrained_layout=False)
plt.subplot(111), plt.imshow(image, "gray"), plt.title("Original greyscale image")
plt.show()

# Display DCT image
plt.figure()
plt.imshow(dct_image, cmap='gray')
plt.title("2D DCT of the image")
plt.show()

# Display DCT watermarked image
plt.figure()
plt.imshow(dct_watermarked, cmap='gray')
plt.title("2D DCT of the image, with watermark")
plt.show()

# Display IDCT watermarked image
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

original = image.copy()
mystery_img = idct_image.copy()

dct_mystery = dct_apply(mystery_img)

averages_mystery = klargest(dct_mystery)
print('-----------------------')
average_original = klargest(mystery_img)

wi_sequence = estimate_mark(average_original, averages_mystery)

gamma_val = gamma(w_sequence, wi_sequence)
print('gamma value  = ', gamma_val)
