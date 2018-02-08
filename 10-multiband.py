import numpy
import cv2

# this exercise references "Pyramid Methods in Image Processing" by Adelson et al.

numpyFirst = cv2.imread(filename='./samples/multiband-apple.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0
numpySecond = cv2.imread(filename='./samples/multiband-orange.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

# blend the apple and the orange using multiband blending with laplacian pyramids
# creating a laplacian pyramid with seven levels for each of the two images

numpyFirst = [ numpyFirst ]
numpySecond = [ numpySecond ]
numpyCombined = []

for intLevel in range(6):
	numpyFirst.append(cv2.pyrDown(numpyFirst[-1]))
	numpySecond.append(cv2.pyrDown(numpySecond[-1]))

	numpyFirst[-2] -= cv2.pyrUp(numpyFirst[-1])
	numpySecond[-2] -= cv2.pyrUp(numpySecond[-1])


# combine the two laplacian pyramids and create a new laplacian pyramid to blend the two images
# specifically, take the left half from numpyFirst and the right half from numpySecond at each level
# afterwards, collapse numpyPyramid to obtain the blended result and store it in numpyOutput

for i in range(len(numpyFirst)):
	first = numpyFirst[i][:, :int(numpyFirst[i].shape[1] / 2)]
	second = numpySecond[i][:, int(numpySecond[i].shape[1] / 2):]
	numpyCombined.append(numpy.hstack((first,second)))

temp = numpyCombined[6]

for i in range(6,0,-1):
       temp = cv2.add(cv2.pyrUp(temp),numpyCombined[i-1])

numpyOutput = temp

cv2.imwrite(filename='./10-multiband.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
#cv2.waitKey(0)

