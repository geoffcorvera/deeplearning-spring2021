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

# Convolve frame 1 with x & y gradient filters
Ix = signal.convolve2d(frame1a, gx, boundary='symm', mode='same')
Iy = signal.convolve2d(frame1a, gy, boundary='symm', mode='same')
# Calculate temporal derivative
It = frame2a - frame1a

# %%
# Create indices
nrows, ncols = frame1a.shape

out_w = ncols-2
out_h = nrows-2

i0 = np.repeat(np.arange(3), 3).reshape(-1,1)
i1 = np.repeat(np.arange(out_h), out_w).reshape(1,-1)
i = i0 + i1

j0 = np.tile(np.arange(3), 3).reshape(-1,1)
j1 = np.tile(np.arange(out_h), out_w).reshape(1,-1)
j = j0 + j1

ix_vectrz = Ix[i,j].reshape(-1,1)
iy_vectrz = Iy[i,j].reshape(-1,1)
it_vectrz = It[i,j].reshape(-1,1)

# %%
# For each pixel in frame 1, calculate optical flow


# %%
subx = Ix[0:3, 0:3]
suby = Iy[0:3, 0:3]
subt = It[0:3, 0:3]

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

# Steps (to do):
    # Calculate optical flow (2D vector) for each px in frame 1:
        # Calculate temporal derivative: pixel intensity deltas over time (img2 - img1)


# Identify Outputs
    ## Vx (M x N): x-component of optical flow for each pixel of the original image
    ## Vy (M x N): y-component of optical flow for each px of the original image
    ## sqrt(Vx**2 + Vy**2) (M x N): kinda like sobel combination of x & y components
# %%
