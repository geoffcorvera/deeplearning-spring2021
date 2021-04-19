# %%
import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np

img1 = image.imread('data/filter1_img.jpg')
img2 = image.imread('data/filter2_img.jpg')

# %%


def convolve2D(img, kernel, padding, stride=1):
    assert kernel.shape[0] == kernel.shape[1]
    k = kernel.shape[0]
    outw = int(1 + (img.shape[0] - k + 2*padding) / stride)
    outh = int(1 + (img.shape[1] - k + 2*padding) / stride)
    output = np.zeros((outw, outh))

    if (padding != 0):
        img = padImage(img, padding)

    # Iterate through image & multiply kernel
    for y in range(img.shape[1]-k):
        for x in range(img.shape[0]-k):
            output[x, y] = np.sum(img[x:x+k, y:y+k] * kernel)

    return output


def padImage(image, pwidth=1):
    rowpad = np.zeros((1, image.shape[1]))
    res = np.append(image, rowpad, 0)
    res = np.append(rowpad, res, 0)

    colpad = np.zeros((res.shape[0], 1))
    res = np.append(res, colpad, 1)
    res = np.append(colpad, res, 1)
    return res


# Discretized Gaussian
gausKernel3 = np.array([(1, 2, 1), (2, 4, 2), (1, 2, 1)]).reshape(3, 3) / 16
gausKernel5 = np.array([(1, 4, 7, 4, 1),
                        (4, 16, 26, 16, 4),
                        (7, 26, 41, 26, 7),
                        (4, 16, 26, 16, 4),
                        (1, 4, 7, 4, 1)]).reshape(5, 5) / 273

# Gaussian partial derivatives
gx = np.array([(1, 0, -1), (2, 0, -2), (1, 0, -1)])
gy = np.array([(1, 2, 1), (0, 0, 0), (-1, -2, -1)])

# %% [markdown]
# Gaussian Filters
# %%
res3x3 = convolve2D(img1, gausKernel3, padding=1)
res5x5 = convolve2D(img1, gausKernel5, padding=2)

plt.subplot(1, 3, 1).imshow(img1, cmap='gray')
plt.title('Original')
plt.subplot(1, 3, 2).imshow(res3x3, cmap='gray')
plt.title('Gaussian 3x3 kernel')
plt.subplot(1, 3, 3).imshow(res5x5, cmap='gray')
plt.title('Gaussian 5x5 kernel')
plt.show()

# %% [markdown]
- determine the size of the resulting matrices from convolution
- create matrix w / 3-dimensional elements(for each color channel)
- output/save resulting image

# %%
# TODO
RGB = [convolve2D(img2[:, :, c], gausKernel3, 1) for c in range(3)]
RGB = np.array(RGB)
_, x, y = RGB.shape

res = np.zeros((530, 800, 3))
for i in range(x):
    for j in range(y):
        res[i, j] = RGB[:, i, j]
plt.imshow(res)
# %% [markdown]
# ## Derivative of Gauss Filter
# %%


def dogFilter(X):
    return (convolve2D(X, gx, 1), convolve2D(X, gy, 1))


dgx1, dgy1 = dogFilter(img1)

plt.subplot(1, 3, 1).imshow(img1, cmap='gray')
plt.title('Original')
plt.subplot(1, 3, 2).imshow(dgx1, cmap='gray')
plt.title('DoG Gx')
plt.subplot(1, 3, 3).imshow(dgy1, cmap='gray')
plt.title('DoG Gy')
plt.show()

# %% [markdown]
# Sobel filter

# %%


def sobelFilter(X):
    return np.sqrt(convolve2D(X, gx, 1)**2 + convolve2D(X, gy, 1)**2)


# %%
channels = np.array([img2[:, :, c] for c in range(3)])
res = [sobelFilter(ch) for ch in channels]

plt.subplot(2, 2, 1).imshow(img2)
plt.title('Original')
for i, ch in enumerate(res):
    plt.subplot(2, 2, i+2).imshow(ch)
plt.show()

# %%
sobel1 = sobelFilter(img1)

plt.subplot(1, 2, 1).imshow(img1, cmap='gray')
plt.title('Original')
plt.subplot(1, 2, 2).imshow(sobel1, cmap='gray')
plt.title('Sobel filter')
plt.show()
