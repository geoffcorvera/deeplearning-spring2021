# %% [markdown]
# ## Approach
# 1. Import image data to array
# 2. Create Gaussian filter
# 3. Apply filter
# 4. Display resulting image (array -> jpg)
# 
# ## Import images

# %%
import matplotlib.pyplot as plt 
import matplotlib.image as image 
import numpy as np
img1 = image.imread('filter1_img.jpg')
img2 = image.imread('filter2_img.jpg')
# img1 = image.imread('p1/filter1_img.jpg')
# img2 = image.imread('p1/filter2_img.jpg')

# %% [markdown]
# ### Image details
# - img1: 512x512, 8-Bit image
# - img2: 530x800, 3-channel 8-bit 

# ## Apply Gaussian Blur
# Convolve the discretized Gaussian filters (3x3 & 5x5) with the input images. Use zero-pad width of 1 for 3x3 filter, and width 2 for 5x5, and *stride* = 1.

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
            output[x,y] = np.sum(img[x:x+k, y:y+k] * kernel)
    
    return output
    

def padImage(image, pwidth=1):
    rowpad = np.zeros((1,image.shape[1]))
    res = np.append(image, rowpad, 0)
    res = np.append(rowpad, res, 0)

    colpad = np.zeros((res.shape[0], 1))
    res = np.append(res, colpad, 1)
    res = np.append(colpad, res, 1)
    return res

# %%
# Discretized Gaussian filters
gausFilter3 = np.array([(1,2,1),(2,4,2),(1,2,1)]).reshape(3,3) / 16
gausFilter5 = np.array([(1,4,7,4,1),
                        (4,16,26,16,4),
                        (7,26,41,26,7),
                        (4,16,26,16,4),
                        (1,4,7,4,1)]).reshape(5,5) / 273

res3x3 = convolve2D(img1, gausFilter3, 1)
res5x5 = convolve2D(img1, gausFilter5, 2)

plt.subplot(2,2,1).imshow(img1, cmap='gray')
plt.title('Original')
plt.subplot(2,2,2).imshow(res3x3, cmap='gray')
plt.title('Gaussian 3x3 kernel')
plt.subplot(2,2,3).imshow(res5x5, cmap='gray')
plt.title('Gaussian 5x5 kernel')
# res2 = convolve2D(img2, gausFilter3, 1)
# plt.subplot(2,2,3).imshow(img2)
# plt.subplot(2,2,4).imshow(res2)
plt.show()

# TODO: apply to layers in img2

# %% [markdown]
## Derivative of Gauss

# %%
def dogFilter(X):
    gx = np.array([(1,0,-1),(2,0,-2),(1,0,-1)])
    gy = np.array([(1,2,1),(0,0,0),(-1,-2,-1)])

    return np.sqrt(convolve2D(X, gx, 1)**2 + convolve2D(X, gy, 1)**2)

dog1 = dogFilter(img1)

plt.subplot(1,2,1).imshow(img1, cmap='gray')
plt.subplot(1,2,2).imshow(dog1, cmap='gray')
plt.show()

