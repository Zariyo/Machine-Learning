import numpy as np
import scipy.signal
import skimage.color
from matplotlib import pyplot as plt
from scipy import signal
from skimage.color import rgb2gray

# tworzymy tablice o wymiarach 128x128x3 (3 kanaly to RGB)
# uzupelnioną zerami = kolor czarny
data = np.zeros((128, 128, 3), dtype=np.uint8)


# chcemy zeby obrazek byl czarnobialy,
# wiec wszystkie trzy kanaly rgb uzupelniamy tymi samymi liczbami
# napiszmy do tego funkcje
def draw(img, x, y, color):
    img[x, y] = [color, color, color]


# zamalowanie 4 pikseli w lewym górnym rogu
draw(data, 5, 5, 100)
draw(data, 6, 6, 100)
draw(data, 5, 6, 255)
draw(data, 6, 5, 255)

# rysowanie kilku figur na obrazku
for i in range(128):
    for j in range(128):
        if (i - 64) ** 2 + (j - 64) ** 2 < 900:
            draw(data, i, j, 200)
        elif i > 100 and j > 100:
            draw(data, i, j, 255)
        elif (i - 15) ** 2 + (j - 110) ** 2 < 25:
            draw(data, i, j, 150)
        elif (i - 15) ** 2 + (j - 110) ** 2 == 25 or (i - 15) ** 2 + (j - 110) ** 2 == 26:
            draw(data, i, j, 255)

# konwersja macierzy na obrazek i wyświetlenie

matrix_vertical = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
]

data = skimage.color.rgb2gray(data)

print(data)
# print(matrix)
data_conv_vertical = scipy.signal.convolve2d(data, matrix_vertical)


def rectify(dataim):
    for ix in range(len(dataim)):
        for iy in range(len(dataim[ix])):
            if dataim[ix, iy] < 0:
                dataim[ix, iy] = 0
            if dataim[ix, iy] > 255:
                dataim[ix, iy] = 255
    return dataim


data_conv_vertical_rectified = rectify(data_conv_vertical.copy())

matrix_horizontal = [
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
]

data_conv_horizontal = scipy.signal.convolve2d(data, matrix_horizontal)

data_conv_horizontal_rectified = rectify(data_conv_horizontal.copy())

fig = plt.figure()
plt.gray()
ax1 = fig.add_subplot(231)  # left side
ax2 = fig.add_subplot(232)  # left side
ax3 = fig.add_subplot(233)  # left side
ax4 = fig.add_subplot(234)  # left side
ax5 = fig.add_subplot(235)  # left side
ax6 = fig.add_subplot(236)  # left side
ax1.imshow(data, interpolation='nearest')
ax2.imshow(data_conv_vertical, interpolation='nearest')
ax3.imshow(data_conv_vertical_rectified, interpolation='nearest')
ax4.imshow(data, interpolation='nearest')
ax5.imshow(data_conv_horizontal, interpolation='nearest')
ax6.imshow(data_conv_horizontal_rectified, interpolation='nearest')
plt.show()
