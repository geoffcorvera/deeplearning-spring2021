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


def normalize(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))


Ix = normalize(Ix)
Iy = normalize(Iy)

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
i, j = generateIndices(Ix)
ix_vctd = Ix[i,j]
iy_vctd = Iy[i,j]
it_vctd = It[i,j]

def decoodeCoordinates(index, width):
    return (math.floor(index / width), index % width)


# %%
# VECTORIZE + LOOP Implementation

out_h = frame2a.shape[0]-2
out_w = frame2a.shape[1]-2
Vx = np.zeros((out_h, out_w))
Vy = np.zeros(Vx.shape)

for i in range(ix_v.shape[1]):
    ix = ix_v[:, i].reshape(-1, 1)
    iy = iy_v[:, i].reshape(-1, 1)

    A = np.concatenate((ix, iy), axis=1)
    b = it_v[:, i].reshape(-1, 1)

    # OLS to solve for optical flow V = <vx, vy>
    OpFlow = np.dot(np.dot(A.T, A), -np.dot(A.T, b))

    row = math.floor(i / out_w)
    col = i % (out_w)
    Vx[row, col] = OpFlow[0][0]
    Vy[row, col] = OpFlow[1][0]

# %%
# Display results
fig, (ax_orig, ax_gx, ax_gy, ax_both) = plt.subplots(4, 1, figsize=(8, 20))
ax_orig.imshow(frame1a)
ax_orig.set_title('Original')
ax_orig.set_axis_off()
ax_gx.imshow(Vx)
ax_gx.set_title('Vx')
ax_gx.set_axis_off()
ax_gy.imshow(Vy)
ax_gy.set_title('Vy')
ax_gy.set_axis_off()
ax_both.imshow(np.sqrt(Vx**2 + Vy**2))
ax_both.set_title('Visualize both Vx & Vy')
ax_both.set_axis_off()


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

        of = np.dot(np.linalg.inv(np.dot(A.T, A)), -np.dot(A.T, b))
        Vx[i, j], Vy[i, j] = of[:, 0]
