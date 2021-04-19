# Followed along OpenCV tutorial:
# https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html

# %%
import numpy as np
import cv2 as cv

# %%
img1 = cv.imread('data/SIFT1_img.jpg')
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

img1 = cv.drawKeypoints(gray, kp, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imwrite('sift1_keypoints.jpg', img1)

# %%
img2 = cv.imread('data/SIFT2_img.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

kp2, des2 = sift.detectAndCompute(gray2, None)

img2 = cv.drawKeypoints(gray2, kp2, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imwrite('sift2_keypoints.jpg', img2)