# %%
import numpy as np 
import matplotlib.image as image
from scipy import signal

# Gaussian partial derivatives
gx = np.array([(1, 0, -1),
               (2, 0, -2),
               (1, 0, -1)])
gy = np.array([(1, 2, 1),
               (0, 0, 0),
               (-1, -2, -1)])

# %%
# Import pair of images (Note: examples are already single-channel)
frame1a = image.imread('frame1_a.png')
frame2a = image.imread('frame2_a.png')
assert frame1a.shape == frame2a.shape

Ix = signal.convolve2d(frame1a, gx, boundary='symm', mode='same')
Iy = signal.convolve2d(frame1a, gy, boundary='symm', mode='same')

def opticalFlow(ix, iy, it):
    xx = ix**2
    xy = ix * iy
    yy = iy**2
    xt = ix * it
    yt = iy * it

    invTerm = np.array([(np.sum(xx), np.sum(xy)),
                        (np.sum(xy), np.sum(yy))])
    invTerm = np.linalg.inv(invTerm)

    return np.dot(invTerm, np.array([-np.sum(xt), -np.sum(yt)]).reshape(2,1))

# %%
import matplotlib.pyplot as plt 

fig, (ax_orig, ax_gx, ax_gy) = plt.subplots(3, 1, figsize=(6, 15))
ax_orig.imshow(frame1a)
ax_orig.set_title('Original')
ax_orig.set_axis_off()
ax_gx.imshow(Ix)
ax_gx.set_title('Ix')
ax_gx.set_axis_off()
ax_gy.imshow(Iy)
ax_gy.set_title('Iy')
ax_gy.set_axis_off()

# Identify Inputs
## pair of M x N images

# Convert images to grayscale
# 



# Identify Outputs
## Vx (M x N): x-component of optical flow for each pixel of the original image
## Vy (M x N): y-component of optical flow for each px of the original image
## sqrt(Vx**2 + Vy**2) (M x N): kinda like sobel combination of x & y components