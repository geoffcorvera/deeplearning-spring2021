import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.image as mpimg

# see image tutorial: https://matplotlib.org/2.0.0/users/image_tutorial.html

x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
plt.show()