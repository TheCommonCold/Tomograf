import sys
import random

from kivy.app import App
from kivy.uix.button import Button
from kivy.graphics import *
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.slider import Slider
from kivy.uix.label import Label


from kivy.app import App
from kivy.graphics import Mesh
from array import array
from functools import partial
from math import cos, sin, pi


import kivy.graphics.texture

import numpy as np
import matplotlib.pyplot as plt
import math
import pydicom
from pydicom.data import get_testdata_files

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import warnings

from kivy.config import Config
Config.set('graphics', 'width', '500')
Config.set('graphics', 'height', '500')

warnings.filterwarnings("ignore")


def sinogramWithSkimage(image, theta):
    return radon(image, theta=theta, circle=True)


def backProjectionWithSkimage(sinogram, theta):
    return iradon(sinogram, theta=theta, circle=True)


def diaplyAll(image, sinogram, reconstruction_fbp, normal):
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
    axes[1, 1].imshow(normal, cmap=plt.cm.Greys_r)
    fig.tight_layout()
    plt.show()


def emmiterPosition(alpha, r, displacement):
    x = r * math.cos(math.radians(alpha)) + displacement[0]
    y = r * math.sin(math.radians(alpha)) + displacement[0]
    return [round(x), round(y)]


def sensorPosition(alpha, r, n, theta, displacement):
    result = []
    for i in range(n):
        x = r * math.cos(
            math.radians(alpha) + math.pi - (math.radians(theta) / 2) + i * (math.radians(theta) / (n - 1)))
        y = r * math.sin(
            math.radians(alpha) + math.pi - (math.radians(theta) / 2) + i * (math.radians(theta) / (n - 1)))
        result.append([round(x + displacement[0]), round(y + displacement[0])])
    return result


def wartoscPiksela(image, x, y, srednia, liczbaPikseli):
    if x < 0 or y < 0 or x >= len(image[0]) or y >= len(image):
        return srednia, liczbaPikseli
    srednia += image[y][x]
    liczbaPikseli += 1
    return srednia, liczbaPikseli


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
    srednia, liczbaPikseli = wartoscPiksela(image, x, y, 0, 0)
    # gdy wiodaca OX
    if dx > dy:
        ai = (dy - dx) * 2
        bi = dy * 2
        d = bi - dx
        while x != x2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                x += xi
            srednia, liczbaPikseli = wartoscPiksela(image, x, y, srednia, liczbaPikseli)
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
            srednia, liczbaPikseli = wartoscPiksela(image, x, y, srednia, liczbaPikseli)
    # print(p1,p2,srednia,liczbaPikseli)
    return 0 if liczbaPikseli == 0 else srednia / liczbaPikseli


def dodajDoPiksela(image, x, y, w, normal):
    if x < 0 or y < 0 or x >= len(image[0]) or y >= len(image):
        return
    image[y][x] += w
    normal[y][x] += 1


def dodawanieBresenhama(image, p1, p2, w, normal):
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
    dodajDoPiksela(image, x, y, w, normal)
    # gdy wiodaca OX
    if dx > dy:
        ai = (dy - dx) * 2
        bi = dy * 2
        d = bi - dx
        while x != x2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                x += xi
                dodajDoPiksela(image, x, y, w, normal)

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
            dodajDoPiksela(image, x, y, w, normal)


def normalizeWithOffset(img,offset=(0,0,0,0)):
    min = max = img[0][0]
    #offsety są góra,lewo,dół, prawo
    for i in range(offset[0],img.shape[0]-offset[2]):
        for j in range(offset[1],img.shape[1]-offset[3]):
            max = max if max > img[i][j] else img[i][j]
            min = min if min < img[i][j] else img[i][j]
    print(min, max)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            img[i][j] = (img[i][j] - min) / max
    img=przytijZera(img)
    return img

def normalize(img,percent=1.0):
    hor=int(img.shape[1]*(1-percent))//2
    vert=int(img.shape[0]*(1-percent))//2
    return normalizeWithOffset(img,(hor,vert,hor,vert))

def VerticalFiltering(image, mask):
    if len(mask)%2==0:
        print("Maska powinna mieć nieparzystą długość, filtr nie działa")
        return image
    new = np.zeros(image.shape)
    margines = len(mask) // 2

    for i in range(margines, len(image) - margines):
        for j in range(len(image[i])):
            for k in range(len(mask)):
                new[i][j] += image[i - margines + k][j] * mask[k]
    return new

def pokaz(img):
    plt.imshow(img, cmap=plt.cm.Greys_r)
    plt.show()


def przytijZera(img, offset=(0, 0, 0, 0)):

    for i in range(offset[0], img.shape[0] - offset[2]):
        for j in range(offset[1], img.shape[1] - offset[3]):
            if img[i][j]<0:
                img[i][j]=0
            elif img[i][j]>1:
                img[i][j] = 1
    return img

def obniz(img,n, offset=(0, 0, 0, 0)):
    for i in range(offset[0], img.shape[0] - offset[2]):
        for j in range(offset[1], img.shape[1] - offset[3]):
            img[i][j]-=n
    przytijZera(img)
    return img

def tomographing(image, alphaStep, SensorCount, theta, scaleR):
    r = math.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2) / 2
    r=r*scaleR;
    imageMiddle = (image.shape[0] // 2, image.shape[1] // 2)
    sinogram = []
    emitters = []
    sensors = []
    katy = np.arange(0, 360, alphaStep)
    print('Tworzenie sinogramu')
    for i in katy:
        kolumnaSinogramu = []
        emitterPos = emmiterPosition(i, r, imageMiddle)
        emitters.append(emitterPos)
        sensorPos = sensorPosition(i, r, SensorCount, theta, imageMiddle)
        sensors.append(sensorPos)
        # print('\r{}/360'.format(i + katy[1]-katy[0]), end='')
        for sensor in sensorPos:
            kolumnaSinogramu.append(sredniaBresenhama(image, emitterPos, sensor))
        sinogram.append(kolumnaSinogramu)
    sinogram = np.array(sinogram).T

    pokaz(sinogram)
    # Filtrowanie
    sinogram = normalize(sinogram)
    sinogram = VerticalFiltering(np.array(sinogram), [-3,7,-3])
    sinogram=przytijZera(sinogram)
    pokaz(sinogram)
    print('\nTworzenie rekonstrukcji')
    # Init Rekonstrukcji
    reconstructed = np.zeros(image.shape)
    normal = np.zeros(image.shape)

    # Rekonstrukcja
    for i in range(len(katy)):
        # print('\r{}/360'.format(katy[i] + katy[1]-katy[0]), end='')
        for j, sensor in enumerate(sensors[i]):
            dodawanieBresenhama(reconstructed, emitters[i], sensors[i][j], sinogram[j][i], normal)
    reconstructed = np.array(reconstructed)
    pokaz(reconstructed)
    # Normalizacja
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            reconstructed[i][j] /= normal[i][j]
    normalize(reconstructed,0.8)
    pokaz(reconstructed)
    return sinogram, reconstructed, normal


def tomograf(alphaStep=0.5, SensorCount=181, theta=180,scaleR=1):
    filePath=0
    print(filePath)
    filename = get_testdata_files("CT_small.dcm")[0]
    ds = pydicom.dcmread(filename)
    image=ds.pixel_array
    #image = imread(data_dir + "/phantom.png", as_gray=True)
    # image = imread("./obrazy/Kolo.jpg", as_gray=True)
    #image = rescale(image, scale=0.4, mode='reflect', multichannel=False)
    # plt.imshow(image, cmap=plt.cm.Greys_r)
    # plt.show()
    sinogram, reconstruction, normal = tomographing(image, alphaStep, SensorCount, theta,scaleR)
    diaplyAll(image, sinogram, reconstruction, normal)
    # theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    # sinogram = sinogramWithSkimage(image, theta)
    # reconstruction = backProjectionWithSkimage(sinogram, theta)
    # diaplyAll(image, sinogram, reconstruction, normal)


class TestApp(App):
    def convert(self,image,pic,*largs):
        max = image.max()
        min = image.min()
        image = np.flip(image, 0)
        image = [int((x - min) * 255 / (max - min)) for x in image.ravel()]
        arr=array('B', image)
        pic.blit_buffer(arr, colorfmt='luminance', bufferfmt='ubyte')
        return arr

    def change_mode(self, mode,image, *largs):
        if mode=='scan':
            sinogram, reconstruction, normal = tomographing(image,3, 100, 180,1)
            diaplyAll(image, sinogram, reconstruction, normal)

            pic1 = kivy.graphics.texture.Texture.create(size=(sinogram.shape[1], sinogram.shape[0]))
            self.convert(sinogram,pic1)
            pic2 = kivy.graphics.texture.Texture.create(size=(image.shape[1], image.shape[0]))
            self.convert(image, pic2)
            pic3 = kivy.graphics.texture.Texture.create(size=(image.shape[1], image.shape[0]))
            self.convert(normal, pic3)

            with self.cnvs.canvas:
                Rectangle(texture=pic1, pos=(250 + 10, 270), size=(200, 200))
                Rectangle(texture=pic2, pos=(250 - 200 - 10, 60), size=(200, 200))
                Rectangle(texture=pic3, pos=(250 + 10, 60), size=(200, 200))

    def build(self):

        self.cnvs = Widget()
        with self.cnvs.canvas:
            Color(1, 1, 1)

        filename = get_testdata_files("CT_small.dcm")[0]
        ds = pydicom.dcmread(filename)
        image = ds.pixel_array
        image = imread("./obrazy/Shepp_logan.jpg", as_gray=True)
        image = rescale(image, scale=0.4, mode='reflect', multichannel=False)

        #sliders=BoxLayout(orientation='vertical')
        #sliders.add_widget(Label(text="number of sliders"))
        #sliders.add_widget(Slider(min=30, max=300, value=160))

        self.pic = kivy.graphics.texture.Texture.create(size=(image.shape[1],image.shape[0]))
        self.convert(image,self.pic)

        with self.cnvs.canvas:
            Rectangle(texture=self.pic, pos=(250-200-10, 270), size=(200, 200))
        buttons = BoxLayout(size_hint=(1, None), height=50)
        for mode in ('scan','save'):
            button = Button(text=mode)
            button.bind(on_release=partial(self.change_mode, mode,image))
            buttons.add_widget(button)

        root = BoxLayout(orientation='vertical')
        root.add_widget(self.cnvs)
        #root.add_widget(sliders)
        root.add_widget(buttons)

        return root


if __name__=='__main__':
    TestApp().run()
    #tomograf(0.01, 160, 180,1);