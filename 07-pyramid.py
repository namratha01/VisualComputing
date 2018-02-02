import numpy
import cv2

# this exercise references "The Laplacian Pyramid as a Compact Image Code" by Burt and Adelson
numpyInput = cv2.imread(filename='./samples/lenna.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

# create a laplacian pyramid with four levels as described in the slides as well as in the referenced paper
numpyPyramid = []
temp = numpyInput

gPyr = [numpyInput.copy()]
down1 = cv2.pyrDown(temp)
down2 = cv2.pyrDown(down1)
down3 = cv2.pyrDown(down2)
down4 = cv2.pyrDown(down3)

up1 = cv2.pyrUp(down1)
up2 = cv2.pyrUp(down2)
up3 = cv2.pyrUp(down3)
up4 = cv2.pyrUp(down4)

Lap1 = temp - up1
Lap2 = down1 - up2
Lap3 = down2 - up3
Lap4 = down3 - up4

numpyPyramid.append(Lap1)
numpyPyramid.append(Lap2)
numpyPyramid.append(Lap3)
numpyPyramid.append(down3)

# the following iterates over the levels in numpyPyramid and saves them as an image accordingly
# level four is just a small-scale representation of the original input image anc can be safed as usual
# the value range for the other levels are outside of [0, 1] and a color mapping is applied before saving them

for intLevel in range(len(numpyPyramid)):
	if intLevel == len(numpyPyramid) - 1:
		cv2.imwrite(filename='./07-pyramid-' + str(intLevel + 1) + '.png', img=(numpyPyramid[intLevel] * 255.0).clip(0.0, 255.0).astype(numpy.uint8))

	elif intLevel != len(numpyPyramid) - 1:
		cv2.imwrite(filename='./07-pyramid-' + str(intLevel + 1) + '.png', img=cv2.applyColorMap(src=((numpyPyramid[intLevel] + 0.5) * 255.0).clip(0.0, 255.0).astype(numpy.uint8), colormap=cv2.COLORMAP_COOL))


