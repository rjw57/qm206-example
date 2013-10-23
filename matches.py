import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

start_time = time.clock()

MIN_MATCH_COUNT = 10

img1 = cv2.imread('left.png')  # queryImage
img2 = cv2.imread('right.png') # trainImage

# convert img1/img2 to greyscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

print "loaded images after", time.clock() - start_time, "seconds"

# Initiate feature detector
fd = cv2.FeatureDetector_create('SIFT')

print "initialised detector after", time.clock() - start_time, "seconds"

kp1 = fd.detect(img1)
kp2 = fd.detect(img2)

print "detect features after", time.clock() - start_time, "seconds"

# Initiate feature descriptor
de = cv2.DescriptorExtractor_create('SIFT')

print "initialised descriptor extractor after", time.clock() - start_time, "seconds"

kp1, des1 = de.compute(img1,kp1)
kp2, des2 = de.compute(img2,kp2)

# Initiate descriptor matcher
dm = cv2.DescriptorMatcher_create('FlannBased')

print "initialised descriptor matcher after", time.clock() - start_time, "seconds"

matches = dm.knnMatch(des1, des2, k=2)

print "found matches after", time.clock() - start_time, "seconds"

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

print "pruned matches after", time.clock() - start_time, "seconds"

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

print "extracted points after", time.clock() - start_time, "seconds"

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()

print "found homography after", time.clock() - start_time, "seconds"

h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

print "warped image after", time.clock() - start_time, "seconds"

print "finished after", time.clock() - start_time, "seconds"
