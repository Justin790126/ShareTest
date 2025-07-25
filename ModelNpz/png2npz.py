from PIL import Image
import numpy as np

# Open the image
img = Image.open('lena.png')

# Convert to NumPy array
img_np = np.array(img)

# Save as .npz
np.savez('lena.npz', lena=img_np)