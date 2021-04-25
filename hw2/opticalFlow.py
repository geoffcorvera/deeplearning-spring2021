# %%
import numpy as np 
import matplotlib.image as image
from scipy import signal
import math

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
# Create indices & vectorize per-pixel sys of equations
nrows, ncols = frame1a.shape

out_w = ncols-2
out_h = nrows-2

i0 = np.repeat(np.arange(3), 3).reshape(-1,1)
i1 = np.repeat(np.arange(out_h), out_w).reshape(1,-1)
i = i0 + i1

j0 = np.tile(np.arange(3), 3).reshape(-1,1)
j1 = np.tile(np.arange(out_h), out_w).reshape(1,-1)
j = j0 + j1

# The columns of these vectors are the unraveled 3x3
# square of px surrounding px(i,j)
ix_vectrz = Ix[i,j]
iy_vectrz = Iy[i,j]
it_vectrz = It[i,j]
assert ix_vectrz.shape == iy_vectrz.shape
assert iy_vectrz.shape == it_vectrz.shape


# %%
# Calculate Optical Flow
Vx = np.zeros((out_h, out_w))
Vy = np.zeros(Vx.shape)

for i in range(ix_vectrz.shape[1]):
    ix = ix_vectrz[:,i].reshape(-1,1)
    iy = iy_vectrz[:,i].reshape(-1,1)

    A = np.concatenate((ix, iy), axis=1)
    b = it_vectrz[:,i].reshape(-1,1)

    # OLS to solve for optical flow V = <vx, vy>
    OpticalFlow_ij = np.dot(np.dot(A.T, A), -np.dot(A.T, b))

    row = math.floor(i / out_w)
    col = i % (out_w)
    Vx[row, col] = OpticalFlow_ij[0]
    Vy[row, col] = OpticalFlow_ij[1]


# %%
import matplotlib.pyplot as plt 

# Visualize optical flow results
fig, (ax_orig, ax_gx, ax_gy, ax_both) = plt.subplots(4, 1, figsize=(6, 15))
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

