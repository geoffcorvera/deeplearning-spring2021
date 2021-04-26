# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from scipy import signal
import math

# Gaussian partial derivatives
gx = np.array([(1, 0, -1), (2, 0, -2), (1, 0, -1)])
gy = np.array([(1, 2, 1), (0, 0, 0), (-1, -2, -1)])
gaussian = np.array([(1, 2, 1), (2, 4, 2), (1, 2, 1)]).reshape(3, 3) / 16

# Import pair of images (Note: examples are already single-channel)
frame1a = image.imread('frame1_a.png')
frame2a = image.imread('frame2_a.png')
assert frame1a.shape == frame2a.shape

# Smooth with Gaussian filter
frame1a = signal.convolve2d(frame1a, gaussian, boundary='symm', mode='same')
frame2a = signal.convolve2d(frame2a, gaussian, boundary='symm', mode='same')

# Convolve frame 1 with x & y gradient filters
Ix = signal.convolve2d(frame1a, gx, boundary='symm', mode='same')
Iy = signal.convolve2d(frame1a, gy, boundary='symm', mode='same')
# Calculate temporal derivative
It = frame2a - frame1a


# %%
# Visualize initial convolutions
fig, (ax_orig, ax_gx, ax_gy, ax_it) = plt.subplots(4, 1, figsize=(10, 25))
ax_orig.imshow(frame1a)
ax_orig.set_title('Original')
ax_orig.set_axis_off()
ax_gx.imshow(Ix)
ax_gx.set_title('Ix')
ax_gx.set_axis_off()
ax_gy.imshow(Iy)
ax_gy.set_title('Iy')
ax_gy.set_axis_off()
ax_it.imshow(It)
ax_it.set_title('Temporal derivative')
ax_it.set_axis_off()


# %%
# Loop implementation
Vx = np.zeros((Ix.shape[0]-2, Ix.shape[1]-2))
Vy = np.zeros(Vx.shape)

for i in range(Vx.shape[0]-2):
    for j in range(Vx.shape[1]-2):
        ix = Ix[i:i+3, j:j+3].reshape(-1, 1)
        iy = Iy[i:i+3, j:j+3].reshape(-1, 1)
        A = np.concatenate((ix, iy), axis=1)
        b = It[i:i+3, j:j+3].reshape(-1, 1)

        of = np.dot(np.linalg.pinv(np.dot(A.T, A)), -np.dot(A.T, b))
        Vx[i, j], Vy[i, j] = of[:, 0]

# %%


def normalize(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))


def threshold(X):
    return (X > np.mean(X)) * X


imgx = threshold(Vx)
imgy = threshold(Vy)

imgx = normalize(imgx)
imgy = normalize(imgy)

# Display results
fig, (ax_orig, ax_gx, ax_gy, ax_both) = plt.subplots(4, 1, figsize=(8, 20))
ax_orig.imshow(frame1a)
ax_orig.set_title('Original')
ax_orig.set_axis_off()
ax_gx.imshow(imgx)
ax_gx.set_title('Vx')
ax_gx.set_axis_off()
ax_gy.imshow(imgy)
ax_gy.set_title('Vy')
ax_gy.set_axis_off()
ax_both.imshow(np.sqrt(imgx**2 + imgy**2))
ax_both.set_title('Visualize both Vx & Vy')
ax_both.set_axis_off()

# %% [markdown]
# # Linear Algrebra implementation
# Below is an implementation that attempts to avoid loops. There is an indexing bug that's
# causing the resulting Vx and Vy calculations to rerpeat

# %%
# Vectorize 3x3 patches of our per-pixel sys of equations


def generateIndices(X):
    nrows, ncols = X.shape
    out_w = ncols-2
    out_h = nrows-2

    i0 = np.repeat(np.arange(3), 3).reshape(-1, 1)
    i1 = np.repeat(np.arange(out_h), out_w).reshape(1, -1)

    j0 = np.tile(np.arange(3), 3).reshape(-1, 1)
    j1 = np.tile(np.arange(out_h), out_w).reshape(1, -1)
    return (i0+i1, j0+j1)


# The columns of these vectors are the unraveled 3x3
# square of px surrounding px(i,j)
imask, jmask = generateIndices(Ix)
ix_v = Ix[imask, jmask]
iy_v = Iy[imask, jmask]
it_v = It[imask, jmask]

# %%
xx = np.sum(ix_v**2, axis=0)
xy = np.sum(ix_v * iy_v, axis=0)
yy = np.sum(iy_v**2, axis=0)

a1 = np.stack((xx, xy), axis=1)
a2 = np.stack((xy, yy), axis=1)
A = np.stack((a1, a2), axis=1)
A = np.apply_over_axes(np.linalg.pinv, A, 0)

bx = np.sum(ix_v * it_v, axis=0)
by = np.sum(iy_v * it_v, axis=0)
B = np.stack((-bx, -by), axis=1)

Vx = list()
Vy = list()
# TODO: Replace janky loop implementation with linear algebra
for a, b in zip(A, B):
    of = np.dot(a, b)
    Vx.append(of[0])
    Vy.append(of[1])

Vx = np.array(Vx).reshape(-1, frame1a.shape[1]-2)
Vy = np.array(Vy).reshape(-1, frame1a.shape[1]-2)

# %%
