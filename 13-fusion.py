import numpy
import cv2

# this exercise references "Exposure Fusion" by Mertens et al.

numpyInputs = [
	cv2.imread(filename='./samples/fusion-1.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0,
	cv2.imread(filename='./samples/fusion-2.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0,
	cv2.imread(filename='./samples/fusion-3.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0
]

# use the quality measures to extract a weight map for each image according to section 3.1
# set the weighting exponents to one, thus equaling the contrition of contrast, saturation, and exposedness
# make sure to add a small epsilon to each weight map to avoid divisions by zero in the subsequent step
# normalize the weight maps such that they sum up to one at each pixel as described in section 3.2
# store the three weight maps in the numpyWeights array which will be used below to perform the blending

numpyWeights = []

def compute_weights(images):
    (wc, ws, we) = (1, 1, 1)
    sigma = 0.2
    epsilon = 0.0000001

    weights = []
    weights_sum = numpy.zeros(images[0].shape[:2], dtype=numpy.float32)
    for image_uint in images:
        image = numpy.float32(image_uint)
        W = numpy.ones(image.shape[:2], dtype=numpy.float32)

        # contrast
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(image_gray, cv2.CV_32F)
        W_contrast = numpy.absolute(laplacian) ** wc
        W = numpy.multiply(W, W_contrast)

        # saturation
        W_saturation = image.std(axis=2, dtype=numpy.float32) ** ws
        W = numpy.multiply(W, W_saturation)

        # well-exposedness
        sigma2 = sigma**2
        W_exposedness = numpy.prod(numpy.exp(-((image - 0.5)**2)/(2*sigma2)), axis=2, dtype=numpy.float32) ** we
        W = numpy.multiply(W, W_exposedness)
	
        W = W + epsilon
        weights_sum = numpy.add(W,weights_sum)

        weights.append(W)

    # normalization
    for i in range(len(weights)):
        weights[i] = numpy.divide(weights[i],weights_sum)

    return weights

numpyWeights = compute_weights(numpyInputs)

# creating the laplacian and gaussian pyramids to perform multiband blending
# defining separate functions for this steps makes the code easier to read

def gaussian_pyramid(numpyInput, intLevels):
	numpyPyramid = [ numpyInput ]

	for intLevel in range(intLevels):
		numpyPyramid.append(cv2.pyrDown(numpyPyramid[-1]))
	# end

	return numpyPyramid
# end

def laplacian_pyramid(numpyInput, intLevels):
	numpyPyramid = [ numpyInput ]

	for intLevel in range(intLevels):
		numpyPyramid.append(cv2.pyrDown(numpyPyramid[-1]))

		numpyPyramid[-2] -= cv2.pyrUp(numpyPyramid[-1])
	# end

	return numpyPyramid
# end

numpyInputs = [ laplacian_pyramid(numpyInput, 6) for numpyInput in numpyInputs ]
numpyWeights = [ gaussian_pyramid(numpyWeight, 6) for numpyWeight in numpyWeights ]

# constructing a laplacian pyramid by using the weights from the gaussian pyramid
# eventually obtaining the fused result by recovering the output from the merged pyramid

numpyPyramid = []

for intLevel in range(len(numpyInputs[0])):
	numpyPyramid.append(sum([ numpyInputs[intInput][intLevel] * numpyWeights[intInput][intLevel][:, :, None] for intInput in range(len(numpyInputs)) ]))
# end

numpyOutput = numpyPyramid.pop(-1)

while len(numpyPyramid) > 0:
	numpyOutput = cv2.pyrUp(numpyOutput) + numpyPyramid.pop(-1)
# end

cv2.imwrite(filename='./13-fusion-1.png', img=(numpyWeights[0][0] * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
cv2.imwrite(filename='./13-fusion-2.png', img=(numpyWeights[1][0] * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
cv2.imwrite(filename='./13-fusion-3.png', img=(numpyWeights[2][0] * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
cv2.imwrite(filename='./13-fusion-4.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
