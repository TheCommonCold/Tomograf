import sys
import random
from PySide2 import QtCore, QtWidgets, QtGui
import skimage

import numpy as np
import matplotlib.pyplot as plt
import math

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon


def sinogramWithSkimage(image, theta):
    return radon(image, theta=theta, circle=True)


def backProjectionWithSkimage(sinogram, theta):
    return iradon(sinogram, theta=theta, circle=True)


def diaplyAll(image, sinogram, reconstruction_fbp):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    imkwargs = dict(vmin=-0.2, vmax=0.2)
    axes[0, 0].set_title("Original")
    axes[0, 0].imshow(image, cmap=plt.cm.Greys_r)

    axes[0, 1].set_title("Radon transform\n(Sinogram)")
    axes[0, 1].set_xlabel("Projection angle (deg)")
    axes[0, 1].set_ylabel("Projection position (pixels)")
    axes[0, 1].imshow(sinogram, cmap=plt.cm.Greys_r,
                      extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

    axes[1, 0].set_title("Reconstruction\nFiltered back projection")
    axes[1, 0].imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    axes[1, 1].set_title("Reconstruction error\nFiltered back projection")
    axes[1, 1].imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
    fig.tight_layout()
    plt.show()


def emmiterPosition(alpha, r,displacement):
    x = r * math.cos(math.radians(alpha))+displacement[0]
    y = r * math.sin(math.radians(alpha))+displacement[0]
    return [round(x), round(y)]


def sensorPosition(alpha, r, n, theta,displacement):
    result = []
    for i in range(n):
        x = r * math.cos(
            math.radians(alpha) + math.pi - (math.radians(theta) / 2) + i * (math.radians(theta) / (n - 1)))
        y = r * math.sin(
            math.radians(alpha) + math.pi - (math.radians(theta) / 2) + i * (math.radians(theta) / (n - 1)))
        result.append([round(x+displacement[0]), round(y+displacement[0])])
    return result


def wartoscPiksela(image, x, y):
    if x < 0 or y < 0 or x >= len(image[0]) or y >= len(image):
        return 0, 0
    return image[y][x], 1


def sredniaBresenhama(image, p1, p2):
    # zmienne
    x1 = x = p1[0]
    y1 = y = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    # ustalanie kierunku x
    xi = 1
    dx = x2 - x1
    if x1 > x2:
        xi = -1
        dx = -dx
    # ustalanie kierunku y
    yi = 1
    dy = y2 - y1
    if y1 > y2:
        yi = -1
        dy = -dy
    w, k = wartoscPiksela(image, x, y)
    srednia = w
    liczbaPikseli = k
    # gdy wiodaca OX
    if dx > dy:
        ai = (dy - dx) * 2
        bi = dy * 2;
        d = bi - dx
        while x != x2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                x += xi
            w, k = wartoscPiksela(image, x, y)
            srednia += w
            liczbaPikseli += k
    # gdy wiodaca OY
    else:
        ai = (dx - dy) * 2
        bi = dx * 2
        d = bi - dy
        while y != y2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                y += yi
            w, k = wartoscPiksela(image, x, y)
            srednia += w
            liczbaPikseli += k
    return srednia / liczbaPikseli

def dodajDoPiksela(image, x, y,w):
    if x < 0 or y < 0 or x >= len(image[0]) or y >= len(image):
        return
    image[y][x]+=w

def dodawanieBresenhama(image, p1, p2,w):
    # zmienne
    x1 = x = p1[0]
    y1 = y = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    # ustalanie kierunku x
    xi = 1
    dx = x2 - x1
    if x1 > x2:
        xi = -1
        dx = -dx
    # ustalanie kierunku y
    yi = 1
    dy = y2 - y1
    if y1 > y2:
        yi = -1
        dy = -dy
    dodajDoPiksela(image, x, y,w)
    # gdy wiodaca OX
    if dx > dy:
        ai = (dy - dx) * 2
        bi = dy * 2;
        d = bi - dx
        while x != x2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                x += xi
                dodajDoPiksela(image, x, y,w)

    # gdy wiodaca OY
    else:
        ai = (dx - dy) * 2
        bi = dx * 2
        d = bi - dy
        while y != y2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                y += yi
            dodajDoPiksela(image, x, y,w)

def tomographing(image, alphaStep, SensorCount, theta):
    r = math.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)/2
    imageMiddle=(image.shape[0]//2,image.shape[1]//2)
    sinogram=[]
    emitters=[]
    sensors=[]
    katy=range(0, 360, alphaStep)
    print('Tworzenie sinogramu')
    for i in katy:
        kolumnaSinogramu=[]
        emitterPos = emmiterPosition(i, r,imageMiddle)
        emitters.append(emitterPos)
        sensorPos = sensorPosition(i, r, SensorCount, theta,imageMiddle)
        sensors.append(sensorPos)
        print('\r{}/360'.format(i+1), end='')
        # print(emmiterPos)
        # print(sensorPos)
        for sensor in sensorPos:
            kolumnaSinogramu.append(sredniaBresenhama(image,emitterPos,sensor))
        sinogram.append(kolumnaSinogramu)
    sinogram=np.array(sinogram)
    plt.imshow(sinogram.T, cmap=plt.cm.Greys_r)
    plt.show()
    print('\nTworzenie rekonstrukcji')
    reconstructed=[]
    for i in range(image.shape[1]):
        reconstructed.append([])
        for j in range(image.shape[0]):
            reconstructed[i].append(0)
    for i in range(len(katy)):
        print('\r{}/360'.format(katy[i] + 1), end='')
        for j,sensor in enumerate(sensors[i]):
            dodawanieBresenhama(reconstructed,emitters[i],sensor,sinogram[i][j])
    plt.imshow(reconstructed, cmap=plt.cm.Greys_r)
    plt.show()


if __name__ == '__main__':
    image = imread(data_dir + "/phantom.png", as_gray=True)
    # image = imread("./Kwadraty2.png", as_gray=True)
    image = rescale(image, scale=0.4, mode='reflect', multichannel=False)
    plt.imshow(image, cmap=plt.cm.Greys_r)
    plt.show()
    tomographing(image, 1, 90, 1)
    # theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    # sinogram = sinogramWithSkimage(image, theta)
    # reconstruction = backProjectionWithSkimage(sinogram, theta)
    # diaplyAll(image, sinogram, reconstruction)
