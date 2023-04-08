import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


# The Importance of Phase and Amplitude in Images

# Read images
parrot_path = r'C:\Users\97250\PycharmProjects\anat_project\wet1\given_data\parrot.png'
# 3.a - Create the input - a selfie
yours_path =  r'C:\Users\97250\PycharmProjects\anat_project\wet1\given_data\yours.jpg'

parrot = cv.imread(parrot_path)
port = cv.imread(yours_path)
width = int(parrot.shape[1])
height = int(parrot.shape[0])
dim = (width, height)
port = cv.resize(port, dim)

parrot_gray = cv.cvtColor(parrot, cv.COLOR_RGB2GRAY)
port_gray = cv.cvtColor(port, cv.COLOR_RGB2GRAY)
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(parrot, cmap='gray')
plt.title('Parrot')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(port_gray, cmap='gray')
plt.title('Selfie')
plt.axis('off')
plt.show()

# 3.b - Ampitude and phase of inputs

port_amp = np.fft.fftshift(np.abs(np.fft.fft2(port_gray)))
port_phase = np.fft.fftshift(np.angle(np.fft.fft2(port_gray)))
parrot_amp = np.fft.fftshift(np.abs(np.fft.fft2(parrot_gray)))
parrot_phase = np.fft.fftshift(np.angle(np.fft.fft2(parrot_gray)))
plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.imshow(np.log(parrot_amp), cmap='gray')
plt.title('Parrot Amplitude')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(np.log(port_amp), cmap='gray')
plt.title('Selfie Amplitude')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(parrot_phase, cmap='gray')
plt.title('Parrot Phase')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(port_phase, cmap='gray')
plt.title('Selfie Phase')
plt.axis('off')
plt.show()


# 3.c - Mixing things up
# 1.An image with the amplitude of yours and the phase of parrot
new_image1 = port_amp * np.exp(1j * parrot_phase)
new_image1 = np.abs(np.fft.ifft2(np.fft.fftshift(new_image1)))
# 2.An image with the amplitude of parrot and the phase of yours
new_image2 = parrot_amp * np.exp(1j * port_phase)
new_image2 = np.abs(np.fft.ifft2(np.fft.fftshift(new_image2)))

plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(new_image1, cmap='gray')
plt.title('An image with the amplitude of Selfie and the phase of parrot')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(new_image2, cmap='gray')
plt.title('An image with the amplitude of parrot and the phase of Selfie')
plt.axis('off')
plt.show()


# 3.d - Let's be random
# 1. An image with a random amplitude (consider the range of values you randomly draw - use uniform
# distribution) and the phase of yours.
random_amp = np.random.uniform(low=0, high=100, size=port_amp.shape)
new_image_1 = random_amp * np.exp(1j * port_phase)
new_image_1 = np.abs(np.fft.ifft2(np.fft.fftshift(new_image1)))

# 2.An image with the amplitude of yours and a random phase (again, consider the range of values and use
# uniform distribution).
random_phase = np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=parrot_phase.shape)
new_image2 =  port_amp * np.exp(1j * random_phase)
new_image2 = np.abs(np.fft.ifft2(np.fft.fftshift(new_image2)))

plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(new_image1, cmap='gray')
plt.title('An image with a random amplitude and the phase of Selfie')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(new_image2, cmap='gray')
plt.title('An image with the amplitude of Selfie and a random phase')
plt.axis('off')
plt.show()
