# imports for the HW
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


# 2.a - Create the input - an image of a building
# TODO - udate the pic f building
building_path = r'C:\Users\97250\PycharmProjects\anat_project\wet1\given_data\building.png'
building = cv.imread(building_path)

# Convert from RGB to gray
building = cv.cvtColor(building, cv.COLOR_BGR2RGB)
building_gray = cv.cvtColor(building, cv.COLOR_RGB2GRAY)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(building, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(building_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# 2.b - 2D-DFT
# Compute the 2D-DFT
twod_fft = np.fft.fft2(building_gray)
# Shift the zero-frequency component to the center of the spectrum
twod_fft = np.fft.fftshift(twod_fft)
# Compute the amplitude
amp_twod_fft = np.log(np.abs(twod_fft))
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(building_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(amp_twod_fft, cmap='gray')
plt.title('Grayscale Image Amplitude')
plt.axis('off')
plt.show()


# 2.c- Low pass frequency filtering
# Create the low frequencies mask
center_l = int(building_gray.shape[0] / 2)
center_k = int(building_gray.shape[1] / 2)
mask = np.zeros(twod_fft.shape)
# 1. The lowest 2% of the frequencies in the l direction (with all their frequencies in the k direction)
mask[center_l - int(building_gray.shape[0] / 2):center_l + int(building_gray.shape[0] / 2), center_k - int(building_gray.shape[1] * 0.02 / 2):center_k + int(building_gray.shape[1] * 0.02 / 2)] = 1
# Apply the mask
filtered_twod_fft = twod_fft * mask
# Inverse transform
filtered_twod_fft = np.fft.ifftshift(filtered_twod_fft)
filtered_twod_fft = np.fft.ifft2(filtered_twod_fft)
image = np.abs(filtered_twod_fft)
# Plot the result and the mask
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(image, cmap='gray')
plt.title('filtered Image Amplitude')
plt.axis('off')
plt.show()

# 2. The lowest 2% of the frequencies in the k direction (with all their frequencies in the l direction).
mask = np.zeros(twod_fft.shape)
mask[center_l - int(building_gray.shape[0] * 0.02 / 2):center_l + int(building_gray.shape[0] * 0.02 / 2), center_k - int(building_gray.shape[1] / 2):center_k + int(building_gray.shape[1] / 2)] = 1
# Apply the mask
filtered_twod_fft = twod_fft * mask
# Inverse transform
filtered_twod_fft = np.fft.ifftshift(filtered_twod_fft)
filtered_twod_fft = np.fft.ifft2(filtered_twod_fft)
image = np.abs(filtered_twod_fft)
# Plot the result and the mask
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(image, cmap='gray')
plt.title('filtered Image Amplitude')
plt.axis('off')
plt.show()

# 3. The lowest 2% of the frequencies in the l direction (with all their frequencies in the k direction)
# and the lowest 2% of the frequencies in the k direction (with all their frequencies in the l direction)
mask = np.zeros(twod_fft.shape)
mask[center_l - int(building_gray.shape[0] * 0.02 / 2):center_l + int(building_gray.shape[0] * 0.02 / 2), center_k - int(building_gray.shape[1] / 2):center_k + int(building_gray.shape[1] / 2)] = 1
mask[center_l - int(building_gray.shape[0] / 2):center_l + int(building_gray.shape[0] / 2), center_k - int(building_gray.shape[1] * 0.02 / 2):center_k + int(building_gray.shape[1] * 0.02 / 2)] = 1
# Apply the mask
filtered_twod_fft = twod_fft * mask
# Inverse transform
filtered_twod_fft = np.fft.ifftshift(filtered_twod_fft)
filtered_twod_fft = np.fft.ifft2(filtered_twod_fft)
image_LPF = np.abs(filtered_twod_fft)
# Plot the result and the mask
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(image_LPF, cmap='gray')
plt.title('filtered Image Amplitude')
plt.axis('off')
plt.show()


# 2.d - Max pass frequency filtering
def max_freq_filtering(fshift, precentege):
    """
    Reconstruct an image using only its maximal amplitude frequencies.
    :param fshift: The fft of an image, **after fftshift** -
    complex float ndarray of size [H x W].
    :param precentege: the wanted precentege of maximal frequencies.
    :return:
    fMaxFreq: The filtered frequency domain result -
    complex float ndarray of size [H x W].
    imgMaxFreq: The filtered image - real float ndarray of size [H x W].
    """
    center_l = int(fshift.shape[0] / 2)
    center_k = int(fshift.shape[1] / 2)
    fMaxFreq = np.ones(fshift.shape)
    for i in range(fMaxFreq.shape[0]):
        for j in range(fMaxFreq.shape[1]):
            if (i - center_l) ** 2 + (j - center_k) ** 2 > (fMaxFreq.shape[0] / 2) ** 2 * precentege:
                fMaxFreq[i, j] = 0
    # Apply the mask fMaxFreq to the fshift
    fshift = fshift * fMaxFreq
    # Inverse transform
    f_ishift = np.fft.ifftshift(fshift)
    imgMaxFreq = np.fft.ifft2(f_ishift)
    imgMaxFreq = np.abs(imgMaxFreq)
    return fMaxFreq, imgMaxFreq


fMaxFreq, imgMaxFreq = max_freq_filtering(twod_fft, 0.1)
# Plot the result
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(imgMaxFreq, cmap='gray')
plt.title('Max-pass filtered using - 10 precent of maximal frequencies')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(building_gray, cmap='gray')
plt.title(' Gray Image ')
plt.axis('off')
plt.show()

plt.style.use('dark_background')
plt.scatter(imgMaxFreq[1], imgMaxFreq[0], color='white')
plt.title('The non-zero frequencies')
plt.axis('on')
plt.show()

# 2.e - Comparison - max frequencies vs. low frequencies
fMaxFreq, imgMaxFreq = max_freq_filtering(twod_fft, 0.04)
# Plot the result
plt.style.use('default')
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(imgMaxFreq, cmap='gray')
plt.title('Max-pass filtered using - 4 precent of maximal frequencies ')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(image_LPF, cmap='gray')
plt.title(' Low-pass filtered Image ')
plt.axis('off')
plt.show()


# 2.f - Influence of the max frequencies precentege
# MSE calculation for one value of P
def mse_calc(pic, percentage: float):
    fMaxFreq, imgMaxFreq = max_freq_filtering(np.fft.fftshift(np.fft.fft2(pic)), percentage)
    MSE = np.zeros(pic.shape)
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            MSE[i, j] = (pic[i, j] - imgMaxFreq[i, j]) ** 2
    MSE = np.sum(MSE) / (pic.shape[0] * pic.shape[1])
    return MSE


# Calculate for P in range 1 to 100
x = np.arange(1, 100, 1)
y = np.zeros(100)
for i in x:
    MSE = mse_calc(building_gray, i / 100)
    y[i] = MSE
plt.figure()
plt.plot(x, y[1:100])
plt.xlabel('Percentage of maximal frequencies')
plt.ylabel('MSE')
plt.xlim(-5, 100)
plt.xticks(np.arange(0, 101, 20))
plt.title('MSE as a function of the percentage of maximal frequencies')
plt.grid()
plt.show()


