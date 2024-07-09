import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

input_image_path = r'\yann_lecun.jpg'
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

resized_image = cv2.resize(image, (600, 600))

f_transform = fft2(resized_image)
f_shifted = fftshift(f_transform)

phase_spectrum = np.angle(f_shifted)

rows, cols = resized_image.shape
center_row, center_col = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)
mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 1

f_shifted_masked = f_shifted * mask
f_ishifted = ifftshift(f_shifted_masked)
image_filtered = np.abs(ifft2(f_ishifted))

def low_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1
    return mask

def high_pass_filter(shape, cutoff):
    return 1 - low_pass_filter(shape, cutoff)

def band_pass_filter(shape, low_cutoff, high_cutoff):
    lp = low_pass_filter(shape, high_cutoff)
    hp = high_pass_filter(shape, low_cutoff)
    return lp * hp

def band_stop_filter(shape, low_cutoff, high_cutoff):
    return 1 - band_pass_filter(shape, low_cutoff, high_cutoff)

def apply_filter(image, filter_mask):
    f_transform = fft2(image)
    f_shifted = fftshift(f_transform)
    filtered_spectrum = f_shifted * filter_mask
    f_ishifted = ifftshift(filtered_spectrum)
    filtered_image = np.abs(ifft2(f_ishifted))
    return filtered_image

cutoffs = [30, 60, 90]
filtered_images = {}
filter_masks = {}

for cutoff in cutoffs:
    lp_mask = low_pass_filter(resized_image.shape, cutoff)
    hp_mask = high_pass_filter(resized_image.shape, cutoff)
    bp_mask = band_pass_filter(resized_image.shape, cutoff // 2, cutoff)
    bs_mask = band_stop_filter(resized_image.shape, cutoff // 2, cutoff)
    
    filter_masks[f'LPF {cutoff}'] = lp_mask
    filter_masks[f'HPF {cutoff}'] = hp_mask
    filter_masks[f'BPF {cutoff // 2}-{cutoff}'] = bp_mask
    filter_masks[f'BSF {cutoff // 2}-{cutoff}'] = bs_mask
    
    filtered_images[f'LPF {cutoff}'] = apply_filter(resized_image, lp_mask)
    filtered_images[f'HPF {cutoff}'] = apply_filter(resized_image, hp_mask)
    filtered_images[f'BPF {cutoff // 2}-{cutoff}'] = apply_filter(resized_image, bp_mask)
    filtered_images[f'BSF {cutoff // 2}-{cutoff}'] = apply_filter(resized_image, bs_mask)


fig, axes = plt.subplots(4, len(cutoffs) + 1, figsize=(20, 20))

axes[0, 0].imshow(resized_image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[1, 0].imshow(image_filtered, cmap='gray')
axes[1, 0].set_title('Filtered Image')
axes[2, 0].imshow(phase_spectrum, cmap='gray')
axes[2, 0].set_title('Phase Spectrum')

for i, cutoff in enumerate(cutoffs):
    axes[0, i + 1].imshow(filtered_images[f'LPF {cutoff}'], cmap='gray')
    axes[0, i + 1].set_title(f'LPF {cutoff}')
    axes[1, i + 1].imshow(filtered_images[f'HPF {cutoff}'], cmap='gray')
    axes[1, i + 1].set_title(f'HPF {cutoff}')
    axes[2, i + 1].imshow(filtered_images[f'BPF {cutoff // 2}-{cutoff}'], cmap='gray')
    axes[2, i + 1].set_title(f'BPF {cutoff // 2}-{cutoff}')
    axes[3, i + 1].imshow(filtered_images[f'BSF {cutoff // 2}-{cutoff}'], cmap='gray')
    axes[3, i + 1].set_title(f'BSF {cutoff // 2}-{cutoff}')

plt.tight_layout()
plt.show()
