#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt

def create_mask(size=128, line_width=8, num_lines=3):
    mask = np.zeros((size, size), dtype=complex)
    spacing = size // (num_lines+1)
    for i in range(num_lines):
        start = (i + 1) * spacing - line_width //2
        end = start + line_width
        mask[:, start : end] = 1
    return mask

mask = create_mask()
# plt.figure(figsize=(5,5))
# plt.title("Mask (Lines)")
# plt.imshow(np.abs(mask), cmap='gray')
# plt.show()


wavelength = 0.193 # um
na = 0.9
sigma = 0.7
pixel_size = 0.05 #um
size = 128

def get_frequency_coordinates(size, pixel_size):
    freq = np.fft.fftfreq(size, d=pixel_size)
    fx, fy = np.meshgrid(freq, freq)
    f_radius = np.sqrt(fx**2, fy**2)
    return fx, fy, f_radius

def pupil_function(f_radius, na, wavelength):
    cutoff = na / wavelength
    return (f_radius <= cutoff).astype(np.float)

def source_function(f_radius, na, wavelength, sigma):
    source_radius = sigma * na / wavelength
    return (f_radius <= source_radius).astype(np.float)

def compute_aerial_image(mask, wavelength, na, sigma, pixel_size):
    size = mask.shape[0]
    # Step 1: Fourier transform of the mask
    mask_ft = np.fft.fft2(mask)
    mask_ft = np.fft.fftshift(mask_ft)
    # Step 2: Frequency coordinates
    fx, fy, f_radius = get_frequency_coordinates(size, pixel_size)
    print(fx, fy, f_radius)
    # Step 3: Define pupil and source
    pupil = pupil_function(f_radius, na, wavelength)
    source = source_function(f_radius, na, wavelength, sigma)

    # TCC kernel = mask pattern * pupil function (freqemcy domain)
    H = pupil * mask_ft

    aerial = np.fft.ifft2(np.fft.ifftshift(H))
    intensity = np.abs(aerial)**2

    return intensity

aerial_image = compute_aerial_image(mask, wavelength, na, sigma, pixel_size)


# Visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Mask")
plt.imshow(np.abs(mask), cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Aerial Image")
plt.imshow(aerial_image, cmap='gray')
plt.show()

