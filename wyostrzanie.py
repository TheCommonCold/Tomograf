import numpy as np

def przytnij(img, offset=(0, 0, 0, 0)):

    for i in range(offset[0], img.shape[0] - offset[2]):
        for j in range(offset[1], img.shape[1] - offset[3]):
            if img[i][j]<0:
                img[i][j]=0
            elif img[i][j]>1:
                img[i][j] = 1
    return img

def splot(image, mask):
    if mask.shape[0]%2==0 or mask.shape[1]%2==0:
        print("Maska powinna mieć nieparzystą długość, filtr nie działa")
        return image
    new = np.zeros(image.shape)
    margines = [0,0]
    for i in range(2):
        margines[i]=mask.shape[i]//2
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i<margines[0] or i+margines[0]>=image.shape[0] or j<margines[1] or j+margines[1]>=image.shape[1]:
                new[i][j] = image[i][j]
            else:
                for k in range(mask.shape[0]):
                    for l in range(mask.shape[1]):
                        new[i][j] += image[i+k-margines[0]][j+l-margines[1]] * mask[k][l]
    return przytnij(new)

def wyostrz(img):
    return splot(img,np.array([[0,-1,0],[-1,5,-1],[0,-1,-0]]))


def boxBlur(img):
    return splot(img,1/9*np.array([[1,1,1],[1,1,1],[1,1,1]]))

def gaussianBlur(img,n=5):
    if n==3:
        return splot(img, 1 / 16 * np.array([[1, 2, 1], [2,4,2], [1, 2, 1]]))
    elif n==5:
        return splot(img, 1 / 256 * np.array([[1, 4,6,4, 1], [4,16,24,16,4], [6,24,36,24,6],[4,16,24,16,4],[1, 4,6,4, 1]]))

def unsharpMasking(img):
    return splot(img, -1 / 256 * np.array(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, -476, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]))
