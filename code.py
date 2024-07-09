import numpy as np
import matplotlib.pyplot as plt
import cv2

# فیلترهای لایه اول
first_layer_filters = {
    'Edge Enhancement': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    'Horizontal Edge Detection': np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
    'Vertical Edge Detection': np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
}

# فیلترهای لایه دوم
second_layer_filters = {
    'Gaussian Blur': np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]) / 273,
    'Large Edge Enhancement': np.array([[0,  0, -1,  0,  0], [0, 0, -1, 0,  0], [-1, -1, 8, -1, -1], [0, 0, -1, 0,  0], [0,  0, -1,  0,  0]])
}

# بارگذاری تصویر
image_path = r'C:\Users\parmiss\Desktop\University\Signal\Final Project_Image Processing\yann_lecun.jpg'
original_image = cv2.imread(image_path)
plt.imshow(original_image)
plt.title('Original Image')
plt.show()

# تبدیل به RGB
rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_image)
plt.title('RGB Image')
plt.show()

# تغییر اندازه
resized_rgb_image = cv2.resize(rgb_image, (600, 600))
plt.imshow(resized_rgb_image)
plt.title('Resized RGB Image')
plt.show()

# تابع اعمال کانولوشن
def apply_convolution(image, kernel, stride=1, padding='valid'):
    kernel = np.flipud(np.fliplr(kernel))
    image_height, image_width, image_channels = image.shape
    kernel_height, kernel_width = kernel.shape
    output_height = int((image_height - kernel_height) / stride + 1)
    output_width = int((image_width - kernel_width) / stride + 1)
    output = np.zeros((output_height, output_width, image_channels))
    
    if padding == 'same':
        pad = (kernel_height - 1) // 2
        imagePadded = np.zeros((image_height + pad*2, image_width + pad*2, image_channels))
        imagePadded[pad:-pad, pad:-pad, :] = image
    else:
        imagePadded = image    
    
    for y in range(0, output_width, stride):
        if y > (image_width - kernel_width):
            break
        for x in range(0, output_height, stride):
            if x > (image_height - kernel_height): 
                 break
            for c in range(image_channels):
                output[x, y, c] = np.sum(imagePadded[x:x+kernel_height, y:y+kernel_width, c] * kernel)
    
    return output

# اعمال فیلترهای لایه اول
fig, axes = plt.subplots(3, 1, figsize=(10, 15))
feature_map1 = []
for i, (filter_name, kernel) in enumerate(first_layer_filters.items()):
    updated_version = apply_convolution(resized_rgb_image, kernel)
    feature_map1.append(updated_version)
    axes[i].imshow(np.clip(updated_version, 0, 255).astype(np.uint8))
    axes[i].set_title(filter_name)
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# استک کردن feature map
feature_map1_3d = np.stack(feature_map1, axis=-1)

# اعمال فیلترهای لایه دوم
fig, axes = plt.subplots(2, 3, figsize=(10, 15))
feature_map2 = []
for i, (filter_name, kernel) in enumerate(second_layer_filters.items()):
    for j, first_layer_result in enumerate(feature_map1):
        updated_version = apply_convolution(first_layer_result, kernel)
        feature_map2.append(updated_version)
        ax = axes[i, j]
        ax.imshow(np.clip(updated_version, 0, 255).astype(np.uint8))
        ax.set_title(filter_name)
        ax.axis('off')
plt.tight_layout()
plt.show()

# استک کردن feature map دوم
feature_map2_3d = np.stack(feature_map2, axis=-1)

# تابع max pooling
def max_pooling(image, size, stride):
    image_height, image_width, image_channels = image.shape
    output_height = (image_height - size) // stride + 1
    output_width = (image_width - size) // stride + 1
    pooled_image = np.zeros((output_height, output_width, image_channels))
    for c in range(image_channels):
        pooled_image[:, :, c] = image[:image_height, :image_width, c].reshape(output_height, size, output_width, size).max(axis=(1, 3))
    return pooled_image

# اعمال max pooling
pooled_image = max_pooling(resized_rgb_image, size=4, stride=4)
plt.imshow(pooled_image.astype(np.uint8))
plt.title('Pooled Image')
plt.show()

# تست تابع max pooling
fig, axes = plt.subplots(2, 3, figsize=(10, 15))
features = feature_map2_3d.shape[-1]
for i in range(features):
    second_layer_result = feature_map2_3d[..., i]
    updated_version = max_pooling(second_layer_result, 6, 6)
    ax = axes[i // 3, i % 3]
    ax.imshow(np.clip(updated_version, 0, 255).astype(np.uint8))
    ax.axis('off')
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

input_image_path = r'C:\Users\parmiss\Desktop\University\Signal\Final Project_Image Processing\yann_lecun.jpg'
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

# نمایش نتایج
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
