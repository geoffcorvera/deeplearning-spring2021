# %%
import numpy as np

def generateIndices(X):
    nrows, ncols = X.shape
    out_w = ncols-2
    out_h = nrows-2

    i0 = np.repeat(np.arange(3), 3).reshape(-1,1)
    i1 = np.repeat(np.arange(out_h), out_w).reshape(1,-1)

    j0 = np.tile(np.arange(3), 3).reshape(-1,1)
    j1 = np.tile(np.arange(out_h), out_w).reshape(1,-1)
    return (i0+i1, j0+j1)

# %%
img0 = np.arange(16).reshape(4,4)
img1 = img0 + 1

i, j = generateIndices(img0)


# %%
import matplotlib.pyplot as plt 

def showImgs(X, Y):
    f, (ax0, ax1) = plt.subplots(2,1,figsize=(6,15))
    ax0.imshow(X)
    ax0.set_axis_off()
    ax0.set_title('Frame 1')
    ax1.imshow(Y)
    ax1.set_axis_off()
    ax1.set_title('Frame 2')
    plt.show()

# showImgs(img0, img1)


# %%
from scipy import signal

# Gaussian partial derivatives
gx = np.array([(1, 0, -1),
               (2, 0, -2),
               (1, 0, -1)])
gy = np.array([(1, 2, 1),
               (0, 0, 0),
               (-1, -2, -1)])

Ix = signal.convolve2d(img0, gx, boundary='symm', mode='same')
Iy = signal.convolve2d(img0, gy, boundary='symm', mode='same')

# %%
f, (ax_og, ax_x, ax_y) = plt.subplots(3,1, figsize=(12,30))
ax_og.imshow(img0)
ax_og.set_axis_off()
ax_x.imshow(Ix)
ax_x.set_axis_off()
ax_y.imshow(Iy)
ax_y.set_axis_off()

# %%
import math
# Calculate optical flow

i, j = generateIndices(img0)
_Ix = Ix[i,j]
_Iy = Iy[i,j]
_It = (Iy-Ix)[i,j]

Vx = np.zeros((Ix.shape[0]-2, Ix.shape[1]-2))
Vy = np.zeros(Vx.shape)

out_w = Ix.shape[1]-2
out_h = Ix.shape[0]-2

for ind in range(_Ix.shape[1]):
    ix = _Ix[:,ind].reshape(-1,1)
    iy = _Iy[:,ind].reshape(-1,1)
    b = _It[:,ind].reshape(-1,1)

    A = np.concatenate((ix, iy), axis=1)
    l = np.linalg.inv(np.dot(A.T, A))
    r = - np.dot(A.T, b)
    OpFlow = np.dot(l,r)

    r = math.floor(ind / out_w)
    c = ind % out_w
    Vx[r,c] = OpFlow.squeeze()[0]
    Vy[r,c] = OpFlow.squeeze()[1]

# %%
f, (ax0, ax1) = plt.subplots(2,1)
ax0.imshow(Vx)
ax0.set_axis_off()
ax1.imshow(Vy)
ax1.set_axis_off()
plt.show()
# %%
