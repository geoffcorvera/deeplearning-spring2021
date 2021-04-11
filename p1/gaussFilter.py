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

# %% [markdown]
# ### Image details
# - img1: 512x512, 8-Bit image
# - img2: 530x800, 3-channel 8-bit 

# %%
plt.imshow(img1, cmap='gray')
plt.show()
plt.imshow(img2)
plt.show()

# %% [markdown]
# ## Apply Gaussian Blur
# Convolve the discretized Gaussian filters (3x3 & 5x5) with the input images. Use zero-pad width of 1 for 3x3 filter, and width 2 for 5x5, and *stride* = 1.

# %%
def convolve2D(image, kernel, padding, stride=1):
    outw = 1 + (image.shape[0] - kernel.shape[0] + 2*padding) / stride
    outh = 1 + (image.shape[1] - kernel.shape[1] + 2*padding) / stride
    img = padImage(image, padding)
    
    # TODO: implement discretized conovolution

    return img

def padImage(image, pwidth=1):
    rowpad = np.zeros((1,image.shape[1]))
    res = np.append(image, rowpad, 0)
    res = np.append(rowpad, res, 0)

    colpad = np.zeros((res.shape[0], 1))
    res = np.append(res, colpad, 1)
    res = np.append(colpad, res, 1)
    return res

# Discretized Gaussian filters
gausFilter3 = np.array([(1,2,1),(2,4,2),(1,2,1)]).reshape(3,3) / 16
gausFilter5 = np.array([(1,4,7,4,1),
                        (4,16,26,16,4),
                        (7,26,41,26,7),
                        (4,16,26,16,4),
                        (1,4,7,4,1)]).reshape(5,5) / 273

convolve2D(img1, gausFilter3, 1)
convolve2D(img1, gausFilter5, 2)


