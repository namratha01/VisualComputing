import numpy
import cv2
import math
import tqdm

numpyInput = cv2.imread(filename='./samples/homography-2.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

# estimate the homography matrix between matching points and warp the image using bilinear interpolation
# creating the mapping between the four corresponding points

intSrc = [ [266, 343], [646, 229], [388, 544], [777, 538] ]
intDst = [ [302, 222], [746, 231], [296, 490], [754, 485] ]

# construct the linear homogeneous system of equations
# use a singular value decomposition to solve the system
# in practice, cv2.findHomography can be used for this
# however, do not use this function for this exercise
A = []

def getPerspectiveTransformMatrix(p1, p2):
    matrixIndex = 0
    A = numpy.zeros((8, 9))
    for i in range(0, len(p1)):
        x = p1[i][0]
        y = p1[i][1]
        u = p2[i][0]
        v = p2[i][1]

        A[matrixIndex] = [x, y, 1, 0, 0, 0, -x * u, -y * u, -u]
        A[matrixIndex + 1] = [0, 0, 0, x, y, 1, -x * v, -y * v, -v]

        matrixIndex = matrixIndex + 2

    U,s,V = numpy.linalg.svd(numpy.array(A,numpy.float32))
    matrix = V[-1, :].reshape(3, 3)/V[-1,-1]
    return matrix

numpyHomography = getPerspectiveTransformMatrix(intSrc, intDst)

# use a backward warping algorithm to warp the source
# to do so, we first create the inverse transform
# use bilinear interpolation for resampling
# in practice, cv2.warpPerspective can be used for this
# however, do not use this function for this exercise
numpyHomography = numpy.linalg.inv(numpyHomography)
numpyOutput = numpy.zeros(numpyInput.shape, numpy.float32)

for intY in tqdm.tqdm(range(numpyInput.shape[0])):
    for intX in range(numpyInput.shape[1]):
        numpyDst = numpy.array([ intX, intY, 1.0 ], numpy.float32)
        numpySrc = numpy.matmul(numpyHomography, numpyDst.T)
        numpySrc = numpySrc / numpySrc[2]

        alpha,i = math.modf(numpySrc[1])
        beta,j = math.modf(numpySrc[0])
        i = int(i)
        j = int(j)

        if (numpySrc[0] < 0.0 or numpySrc[0] > numpyOutput.shape[1] - 1.0):
            continue   
        elif (numpySrc[1] < 0.0 or numpySrc[1] > numpyOutput.shape[0] - 1.0): 
            continue
        
        numpyOutput[intY, intX] = (1-alpha)*(1-beta)*numpyInput[i,j] + (1-alpha)*(beta)*numpyInput[i,j+1] + (alpha)*(1-beta)*numpyInput[i+1,j] + (alpha)*(beta)*numpyInput[i+1,j+1]

cv2.imwrite(filename='./08-homography.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
