import numpy
import cv2

numpyInput = cv2.imread(filename='./samples/prokudin.png', flags=cv2.IMREAD_GRAYSCALE).astype(numpy.float32) / 255.0

# align the individual images such that their combination results in a proper color image
# splitting the input into the three individual channels while slightly corrping the boundary

intFirst, intSecond = int(1.0 * numpyInput.shape[0] / 3.0), int(2.0 * numpyInput.shape[0] / 3.0)

numpyB = numpyInput[:intFirst, :][50:-50, 50:-50]
numpyG = numpyInput[intFirst:intSecond, :][50:-50, 50:-50]
numpyR = numpyInput[intSecond:, :][50:-50, 50:-50]

# find the homography matrices between the images using cv2.findTransformECC
(ccBG,hBG) = cv2.findTransformECC(numpyB,numpyG,numpy.eye(3,3,dtype = numpy.float32),cv2.MOTION_HOMOGRAPHY)
(ccBR,hBR) = cv2.findTransformECC(numpyB,numpyR,numpy.eye(3,3,dtype = numpy.float32),cv2.MOTION_HOMOGRAPHY)

# based on these matrices, warp numpyG and numpyR towards numpyB using cv2.warpPerspective
# make sure to use inverse warping as stated in the docuḿentation of cv2.findTransformECC
numpyG = cv2.warpPerspective (numpyG,hBG, (numpyG.shape[1],numpyG.shape[0]), flags=cv2.WARP_INVERSE_MAP)
numpyR = cv2.warpPerspective (numpyR,hBR, (numpyR.shape[1],numpyG.shape[0]), flags=cv2.WARP_INVERSE_MAP)

# once the channels are aligned, they can simply be stacked to create the color image
numpyOutput = numpy.stack([numpyB, numpyG, numpyR], 2)

cv2.imwrite(filename='./09-colorize.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
#cv2.waitKey(delay = 0)
