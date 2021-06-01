import numpy as np
import cv2
import os
import math
import skimage.morphology

from numpy import mean
from skimage.io import imread
from skimage.color import rgb2gray


# Function that selects a good amount of frames given a video
def frames(video, path):
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 0

    while success and count < 250:
        if count % 11 == 0:
            cv2.imwrite(path + "frame%d.jpg" % count, image)
        success, image = vidcap.read()
        count += 1


def mei(directory, end_path):
    listofframes = os.listdir(directory)
    for i in range(len(listofframes)-1):

        path1 = directory + '/' + listofframes[i]
        path2 = directory + '/' + listofframes[i+1]
        framex = rgb2gray(imread(path1))
        framey = rgb2gray(imread(path2))

        diffimg = np.zeros(framex.shape)

        for x in range(0, framex.shape[0]):
            for y in range(0, framex.shape[1]):
                if framex[x][y] != framey[x][y]:
                    diffimg[x][y] = (framey[x][y] - framex[x][y]) * 255

        cv2.imwrite(end_path + str(i) + ".jpg", diffimg)


def morphops(directory, end_path):
    listoffiles = os.listdir(directory)

    # Erosion, Dilation and Closing operation
    for filename in listoffiles:
        img = imread(directory + '/' + filename)

        # Erosion to remove details
        img_erosion = skimage.morphology.erosion(img, skimage.morphology.square(3))
        # Dilatation to smooth the image
        img_dilation = skimage.morphology.dilation(img_erosion,  skimage.morphology.square(3))
        # Closing for fun
        img_closing = skimage.morphology.closing(img_dilation, skimage.morphology.square(3))

        cv2.imwrite(end_path + filename, img_closing)


# Edge detection
def edge(directory, end_path):
    listoffiles = os.listdir(directory)

    for filename in listoffiles:
        img = imread(directory + '/' + filename)

        img_edge = cv2.Canny(img, 255, 150)

        cv2.imwrite(end_path + filename, img_edge)


# Extration of the shape descriptor
def extractshape(directory):
    listoffiles = os.listdir(directory)
    newmom = []

    for filename in listoffiles:
        path = directory + filename
        print(path)
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
        # Compute the 7 moments for each frame
        moment = cv2.moments(img)
        hu_moments = cv2.HuMoments(moment)

        # bring them to the same scale
        for i in range(0, 7):
            newmom.append(-1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i])))


def noise(directory, end_path):
    listofdir = os.listdir(directory)

    for filename in listofdir:
        img = img = imread(directory + '/' + filename)

        means = 0.0
        std = 100
        noisy_img = img + np.random.normal(means, std, img.shape)

        cv2.imwrite(end_path + 'noisy_' + filename, noisy_img)


def denoise(directory, end_path):
    listofdir = os.listdir(directory)

    for filename in listofdir:
        img = cv2.imread(directory + '/' + filename, cv2.IMREAD_GRAYSCALE)
        denoised_img = np.zeros(img.shape)
        denoised_img = cv2.fastNlMeansDenoising(img, denoised_img, 55)

        cv2.imwrite(end_path + '/denoisy_' + filename, denoised_img)


def differences(path1, path2, path3):
    list1 = os.listdir(path1)
    list2 = os.listdir(path2)
    list3 = os.listdir(path3)

    lengths = [len(list1), len(list2), len(list3)]
    lengths.sort()

    averages = []

    for i in range(lengths[0]):

        im1 = imread(path1 + list1[i])
        im2 = imread(path2 + list2[i])
        im3 = imread(path3 + list3[i])

        d2 = cv2.matchShapes(im1, im2, cv2.CONTOURS_MATCH_I2, 0)
        d22 = cv2.matchShapes(im1, im3, cv2.CONTOURS_MATCH_I2, 0)
        d222 = cv2.matchShapes(im2, im3, cv2.CONTOURS_MATCH_I2, 0)

        averages.append((d2 + d22 + d222)/3)

    print('--------------Average difference: ', mean(averages))


# -----------Waving video-----------
print('-----------Waving video-----------')
# Compute MEI
mei('FrameVideoWave', "MEIsWave/MEI")
# Clean the frames
morphops("MEIsWave", "MEIsWaveRefactored/Refactored")
# Edge detection
edge("MEIsWaveRefactored", "MEIsWaveEdge/")
# Compute the Hu moments
extractshape("MEIsWaveEdge/")


# -----------Walking video-----------
print('-----------Walking video-----------')
# Compute MEI
mei('FrameVideoWalk', "MEIsWalk/MEI")
# Clean the frames
morphops("MEIsWalk", "MEIsWalkRefactored/Refactored")
# Edge detection
edge("MEIsWalkRefactored", "MEIsWalkEdge/")
# Compute the Hu moments
extractshape("MEIsWalkEdge/")


# -----------Running video-----------
print('-----------Running video-----------')
# Compute MEI
mei('FrameVideoRun', "MEIsRun/MEI")
# Clean the frames
morphops("MEIsRun", "MEIsRunRefactored/Refactored")
# Edge detection
edge("MEIsRunRefactored", "MEIsRunEdge/")
# Compute the Hu moments
extractshape("MEIsRunEdge/")

differences('MEIsRun/', 'MEIsWalk/', 'MEIsWave/')

# -----------Running Noise video-----------
print('-----------Running Noise video-----------')
# Noise images
# noise('FrameVideoRun', 'FrameVideoRunNoisy')
# Denoise images
# denoise('FrameVideoRunNoisy', 'FrameVideoRunDenoised')
# Compute Noisy MEI
mei('FrameVideoRunNoisy', 'MEIsRunNoisy/MEI')
# Compute Denoised MEI
mei('FrameVideoRunDenoised', "MEIsRunDenoised/MEI")
# Clean the frames
morphops("MEIsRunNoisy", "MEIsRunRefactoredNoisy/Refactored")
morphops("MEIsRunDenoised", "MEIsRunRefactoredDenoised/Refactored")
# Edge detection
edge("MEIsRunRefactoredNoisy", "MEIsRunEdgeNoisy/")
edge("MEIsRunRefactoredDenoised", "MEIsRunEdgeDenoised/")
# Compute the Hu moments
extractshape("MEIsRunEdgeNoisy/")
extractshape("MEIsRunEdgeDenoised/")

differences('MEIsWave/', 'MEIsWalk/', 'MEIsRunNoisy/')
differences('MEIsWave/', 'MEIsWalk/', 'MEIsRunDenoised/')
