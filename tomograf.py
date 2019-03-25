import sys
import random
import skimage

import numpy as np
import matplotlib.pyplot as plt
import math

from skimage.io import imread,imsave
from skimage import data_dir
from skimage.transform import radon, rescale, iradon

from wyostrzanie import przytnij,wyostrz,boxBlur,gaussianBlur,unsharpMasking,wyostrz2
import warnings

test = False
warnings.filterwarnings("ignore")

def bladSredniokwadratowy(img,img2):
    for i in range(2):
        if img.shape[i]!=img2.shape[i]:
            print("Obrazy są róznych wymiarów")
            return -1.567
    blad=0.0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            blad+=((img[i][j]-img2[i][j])**2)/(img.shape[0]*img.shape[1])
    return blad


def sinogramWithSkimage(image, theta):
    return radon(image, theta=theta, circle=True)


def backProjectionWithSkimage(sinogram, theta):
    return iradon(sinogram, theta=theta, circle=True)


def diaplyAll(image, sinogram, reconstruction_fbp, normal):
    if test == False:
        return
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




def Bresenham(p1,p2):
    # zmienne
    lista=[]
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
    lista.append([x,y])
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
            lista.append([x,y])
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
            lista.append([x,y])
    # print(p1,p2,srednia,liczbaPikseli)
    return lista

def wartoscPiksela(image, x, y, srednia, liczbaPikseli,normal):
    if x < 0 or y < 0 or x >= len(image[0]) or y >= len(image):
        return srednia, liczbaPikseli,normal
    srednia += image[y][x]
    liczbaPikseli += 1
    normal[y][x] += 1
    return srednia, liczbaPikseli,normal

def sredniaBresenhama(image, p1, p2,normal):
    lista=Bresenham(p1,p2)
    srednia=liczbaPikseli=0
    for i in lista:
        srednia, liczbaPikseli,normal = wartoscPiksela(image, i[0], i[1], srednia, liczbaPikseli,normal)
    return 0 if liczbaPikseli == 0 else srednia / liczbaPikseli, normal


def dodajDoPiksela(image, x, y, w, normal):
    if x < 0 or y < 0 or x >= len(image[0]) or y >= len(image):
        return
    normal[y][x]+=1
    if normal[y][x]>1:
        image[y][x]=((image[y][x]*(normal[y][x]-1))+w)/normal[y][x]
    else:
        image[y][x] = w


def dodawanieBresenhama(image, p1, p2, w, normal):
   lista=Bresenham(p1,p2)
   for i in lista:
        dodajDoPiksela(image, i[0],i[1], w, normal)

def normalizeWithOffset(img,offset=(0,0,0,0)):
    min = max = img[0][0]
    #offsety są góra,lewo,dół, prawo
    for i in range(offset[0],img.shape[0]-offset[2]):
        for j in range(offset[1],img.shape[1]-offset[3]):
            max = max if max > img[i][j] else img[i][j]
            min = min if min < img[i][j] else img[i][j]
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            img[i][j] = (img[i][j] - min) / max
    img=przytnij(img)
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

    for i in range(0, len(image)):
        for j in range(len(image[i])):
            if i < margines or i >= len(image) - margines:
                new[i][j] = image[i][j]
            else:
                for k in range(len(mask)):
                    new[i][j] += image[i - margines + k][j] * mask[k]
    return new



def pokaz(img):
    if test == False:
        return
    plt.imshow(img, cmap=plt.cm.Greys_r)
    plt.show()



def obniz(img,n, offset=(0, 0, 0, 0)):
    for i in range(offset[0], img.shape[0] - offset[2]):
        for j in range(offset[1], img.shape[1] - offset[3]):
            img[i][j]-=n
    przytnij(img)
    return img

def tomographing(image, alphaStep, SensorCount, theta):
    r = math.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2) / 2
    imageMiddle = (image.shape[0] // 2, image.shape[1] // 2)
    sinogram = []
    emitters = []
    sensors = []
    katy = np.arange(0, 360, alphaStep)
    print('Tworzenie sinogramu')
    normal = np.zeros(image.shape)
    for i in katy:
        kolumnaSinogramu = []
        emitterPos = emmiterPosition(i, r, imageMiddle)
        emitters.append(emitterPos)
        sensorPos = sensorPosition(i, r, SensorCount, theta, imageMiddle)
        sensors.append(sensorPos)
        # print('\r{}/360'.format(i + katy[1]-katy[0]), end='')
        for sensor in sensorPos:
            q,normal=sredniaBresenhama(image, emitterPos, sensor,normal)
            kolumnaSinogramu.append(q)
        sinogram.append(kolumnaSinogramu)
    sinogram = np.array(sinogram).T

    pokaz(sinogram)
    # Filtrowanie
    sinogram = normalize(sinogram)
    sinogram = VerticalFiltering(np.array(sinogram), [-2,5,-2])
    sinogram=przytnij(sinogram)
    pokaz(sinogram)
    print('\nTworzenie rekonstrukcji')
    # Init Rekonstrukcji
    reconstructed = np.zeros(image.shape)
    normal2 = np.zeros(image.shape)
    coIteracje=[]
    # Rekonstrukcja
    for i in range(len(katy)):
        # print('\r{}/360'.format(katy[i] + katy[1]-katy[0]), end='')
        for j, sensor in enumerate(sensors[i]):
            dodawanieBresenhama(reconstructed, emitters[i], sensors[i][j], sinogram[j][i], normal2)
        coIteracje.append(bladSredniokwadratowy(image,reconstructed))
    reconstructed = np.array(reconstructed)
    # Normalizacja
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         reconstructed[i][j] /= normal2[i][j]
    pokaz(image)
    pokaz(reconstructed)

    normalize(reconstructed,0.08)
    pokaz(reconstructed)
    # imsave("org.png", image)
    # imsave("recon.png",reconstructed)
    zwyklyBlad=bladSredniokwadratowy(image,reconstructed)
    filtrowane=[]
    # funkcje=[wyostrz,boxBlur,lambda x:gaussianBlur(x,3),lambda x:gaussianBlur(x,5),unsharpMasking,wyostrz2]
    funkcje = [lambda x: gaussianBlur(x, 3)]
    for f in funkcje:
        q=bladSredniokwadratowy(image, f(reconstructed.copy()))
        filtrowane.append(q)
    filtrowane = filtrowane[0]
    # x=0.05
    # for i in range(math.ceil(1/x)-10):
    #     obniz(reconstructed,x)
    #     pokaz(reconstructed)
    return coIteracje,zwyklyBlad,filtrowane


def tomograf(image,alphaStep=1,sensorCount=181,theta=270):

    if theta<sensorCount:
        sensorCount=theta
    # image = imread(data_dir + "/phantom.png", as_gray=True)

    # plt.imshow(image, cmap=plt.cm.Greys_r)
    # plt.show()
    coIteracje,zwykly,filtry = tomographing(image, alphaStep=alphaStep, SensorCount=sensorCount, theta=theta)
    return coIteracje,zwykly,filtry
    # diaplyAll(image, sinogram, reconstruction, normal)
    # theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    # sinogram = sinogramWithSkimage(image, theta)
    # reconstruction = backProjectionWithSkimage(sinogram, theta)
    # diaplyAll(image, sinogram, reconstruction, normal)

if __name__=='__main__':
    listaNazw=["Shepp_logan","SADDLE_PE","CT_ScoutView","Kolo","Paski2","Kropka"]

    for i in listaNazw:
        image = imread("./obrazy/" + i+".jpg", as_gray=True)
        image = rescale(image, scale=0.25, mode='reflect', multichannel=False)
        coIteracje,zwykly,filtry = tomograf(image.copy())
        file = open(i+'.txt', 'w')
        file.write(str(zwykly) + '\n')
        file.write(str(filtry) + '\n')
        for i in coIteracje:
            file.write(str(i)+'\n')
        file.write("-1\n")
        for j in [0.5,1,1.5,2,5,10,15,20]:
            coIteracje, zwykly, filtry = tomograf(image.copy(),alphaStep=j)
            file.write(str(zwykly) + '\n')
            print(j)
            if test:
                break
        file.write("-2\n")
        for j in [15,31,41,51,61,71,81,91,101,111,141,161,181,221,241,261,281,301]:
            coIteracje, zwykly, filtry = tomograf(image.copy(),sensorCount=j,alphaStep=1)
            file.write(str(zwykly) + '\n')
            print(j)
            if test:
                break
        file.write("-3\n")
        for j in [30,40,50,60,70,80,90,100,110,120,140,160,180,200,220,240,270]:
            coIteracje, zwykly, filtry = tomograf(image.copy(), theta=j,alphaStep=1)
            file.write(str(zwykly) + '\n')
            print(j)
            if test:
                break
        file.write("-4\n")
        file.close()
        print(i)
    # image = imread("org.png", as_gray=True)
    # recon = imread("recon.png", as_gray=True)
    # print(bladSredniokwadratowy(image,recon))