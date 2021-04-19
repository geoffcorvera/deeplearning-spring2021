# https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html

# %%
import numpy as np
import cv2 as cv

# %%
img1 = cv.imread('data/SIFT1_img.jpg')
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
keypoints = sift.detect(gray, None)

img1 = cv.drawKeypoints(gray, keypoints, img1)

cv.imwrite('sift1_keypoints.jpg', img1)

# %%
img2 = cv.imread('data/SIFT2_img.jpg')
gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# keypoints
