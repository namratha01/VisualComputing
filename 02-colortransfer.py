import numpy
import cv2

# this exercise references "Color Transfer between Images" by Reinhard et al.

numpyFrom = cv2.imread(filename='./samples/transfer-from.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0
numpyTo = cv2.imread(filename='./samples/transfer-to.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

numpyFrom = cv2.cvtColor(src=numpyFrom, code=cv2.COLOR_BGR2Lab)
numpyTo = cv2.cvtColor(src=numpyTo, code=cv2.COLOR_BGR2Lab)

# match the color statistics of numpyTo to those of numpyFrom

# calculate the per-channel mean of the data points / pixels of numpyTo, and subtract these from numpyTo according to equation 10
# calculate the per-channel std of the data points / pixels of numpyTo and numpyFrom, and scale numpyTo according to equation 11
# calculate the per-channel mean of the data points / pixels of numpyFrom, and add these to numpyTo according to the description after equation 11
b1,g1,r1 = cv2.split(numpyTo)
b2,g2,r2 = cv2.split(numpyFrom)

b1mean = numpy.mean(b1)
b2mean = numpy.mean(b2)
g1mean = numpy.mean(g1)
g2mean = numpy.mean(g2)
r1mean = numpy.mean(r1)
r2mean = numpy.mean(r2)

b1To = numpy.subtract(b1,b1mean)
g1To = numpy.subtract(g1,g1mean)
r1To = numpy.subtract(r1,r1mean)

b1std = numpy.std(b1)
b2std = numpy.std(b2)
g1std = numpy.std(g1)
g2std = numpy.std(g2)
r1std = numpy.std(r1)
r2std = numpy.std(r2)

b1b2To = (b2std/b1std)*b1To
g1g2To = (g2std/g1std)*g1To
r1r2To = (r2std/r1std)*r1To

numpyTob = numpy.add(b2mean,b1b2To)
numpyTog = numpy.add(g2mean,g1g2To)
numpyTor = numpy.add(r2mean,r1r2To)

numpyTo = cv2.merge((numpyTob,numpyTog,numpyTor))

numpyTo[:, :, 0] = numpyTo[:, :, 0].clip(0.0, 100.0)
numpyTo[:, :, 1] = numpyTo[:, :, 1].clip(-127.0, 127.0)
numpyTo[:, :, 2] = numpyTo[:, :, 2].clip(-127.0, 127.0)

numpyOutput = cv2.cvtColor(src=numpyTo, code=cv2.COLOR_Lab2BGR)

cv2.imwrite(filename='./02-colortransfer.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
