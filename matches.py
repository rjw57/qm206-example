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

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

print "detect features after", time.clock() - start_time, "seconds"

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

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
