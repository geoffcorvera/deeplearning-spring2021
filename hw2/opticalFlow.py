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

import matplotlib.pyplot as plt 

# Identify Inputs
## pair of M x N images

# Convert images to grayscale
# 



# Identify Outputs
## Vx (M x N): x-component of optical flow for each pixel of the original image
## Vy (M x N): y-component of optical flow for each px of the original image
## sqrt(Vx**2 + Vy**2) (M x N): kinda like sobel combination of x & y components