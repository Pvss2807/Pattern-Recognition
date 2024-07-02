#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image
img = cv2.imread('/Users/pvss2807/Downloads/P024p2mWOD4BK31HCd3T6K.jpg')

# Convert the image to grayscale as a preprocessing step
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate the power spectrum of the image for simplified analysis
f = np.fft.fft2(gray_img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# Define the Gaussian kernel for smoothening the image
kernel_size = 21
kernel = cv2.getGaussianKernel(kernel_size, 0)
kernel = np.outer(kernel, kernel.transpose())

# Calculate the power spectrum of the kernel for analysis on the smoothened image
f_kernel = np.fft.fft2(kernel, s=gray_img.shape)
fshift_kernel = np.fft.fftshift(f_kernel)
magnitude_spectrum_kernel = 20*np.log(np.abs(fshift_kernel))

# Calculate the Wiener filter
k = 0.1
snr = 1.0
H = f_kernel
H_conj = np.conj(H)
W = H_conj / (H_conj*H + (snr/k**2))
G = fshift * W
g = np.fft.ifft2(np.fft.ifftshift(G))

# Normalize the image and save it
deblurred_img = np.abs(g)
deblurred_img = (deblurred_img - np.min(deblurred_img)) / (np.max(deblurred_img) - np.min(deblurred_img)) * 255
plt.imshow(img)

img = cv2.imread('/Users/pvss2807/Downloads/deblurred.png')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect edges using Canny edge detection
edges = cv2.Canny(gray_img, 100, 200)

# Find contours from the edges
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
contour_img = np.copy(img)
cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)

# Save the contour image
#plt.imshow(contour_img)
#cv2.imwrite('contour_image.jpg', contour_img)


# Read the contour image
contour_img = cv2.imread('/Users/pvss2807/Downloads/contour_img.png')

# Convert the image to grayscale
gray_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)

# Detect circles using Hough transform
circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

# Detect rectangles and squares using the rectangle detection algorithm
rectangles = []
squares = []
edges = cv2.Canny(gray_img, 50, 150)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        x,y,w,h = cv2.boundingRect(cnt)
        if abs(w-h) <= 5:
            squares.append(cnt)
        else:
            rectangles.append(cnt)

# Detect triangles and hexagons using contour detection
triangles = []
hexagons = []
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    if len(approx) == 3:
        triangles.append(cnt)
    elif len(approx) == 6:
        hexagons.append(cnt)

# Draw the detected shapes on the original image
shape_img = np.copy(contour_img)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(shape_img, (x, y), r, (0, 0, 255), 2)
# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         cv2.circle(shape_img, (i[0], i[1]), i[2], (0, 0, 255), 2)
for cnt in squares:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(shape_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
for cnt in rectangles:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(shape_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
for cnt in triangles:
    cv2.drawContours(shape_img, [cnt], 0, (0, 255, 255), 2)
for cnt in hexagons:
    cv2.drawContours(shape_img, [cnt], 0, (255, 255, 0), 2)

# Save the shape image
plt.imshow(shape_img)
print('Number of circles detected:', len(circles))
print('Number of rectangles detected:', len(rectangles))
print('Number of squares detected:', len(squares))
print('Number of triangles detected:', len(triangles))
print('Number of hexagons detected:', len(hexagons))


# In[81]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the blurred image
img = cv2.imread('/Users/pvss2807/Downloads/P034mW_OD4_Bk31HC31d3um_NF.jpg')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate the power spectrum of the image
f = np.fft.fft2(gray_img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# Define the blur kernel (in this case, a Gaussian kernel)
kernel_size = 21
kernel = cv2.getGaussianKernel(kernel_size, 0)
kernel = np.outer(kernel, kernel.transpose())

# Calculate the power spectrum of the kernel
f_kernel = np.fft.fft2(kernel, s=gray_img.shape)
fshift_kernel = np.fft.fftshift(f_kernel)
magnitude_spectrum_kernel = 20*np.log(np.abs(fshift_kernel))

# Calculate the Wiener filter
k = 0.1
snr = 1.0
H = f_kernel
H_conj = np.conj(H)
W = H_conj / (H_conj*H + (snr/k**2))
G = fshift * W
g = np.fft.ifft2(np.fft.ifftshift(G))

# Normalize the image and save it
deblurred_img = np.abs(g)
deblurred_img = (deblurred_img - np.min(deblurred_img)) / (np.max(deblurred_img) - np.min(deblurred_img)) * 255
plt.imshow(img)

img = cv2.imread('/Users/pvss2807/Downloads/deblurred2.png')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect edges using Canny edge detection
edges = cv2.Canny(gray_img, 100, 200)

# Find contours from the edges
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
contour_img = np.copy(img)
cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)

# Save the contour image
#plt.imshow(contour_img)
#cv2.imwrite('contour_image.jpg', contour_img)


# Read the contour image
contour_img = cv2.imread('/Users/pvss2807/Downloads/contour2_img.png')

# Convert the image to grayscale
gray_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)

# Detect circles using Hough transform
circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

# Detect rectangles and squares using the rectangle detection algorithm
rectangles = []
squares = []
edges = cv2.Canny(gray_img, 50, 150)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        x,y,w,h = cv2.boundingRect(cnt)
        if abs(w-h) <= 5:
            squares.append(cnt)
        else:
            rectangles.append(cnt)

# Detect triangles and hexagons using contour detection
triangles = []
hexagons = []
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    if len(approx) == 3:
        triangles.append(cnt)
    elif len(approx) == 6:
        hexagons.append(cnt)

# Draw the detected shapes on the original image
shape_img = np.copy(contour_img)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(shape_img, (x, y), r, (0, 0, 255), 2)
# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         cv2.circle(shape_img, (i[0], i[1]), i[2], (0, 0, 255), 2)
for cnt in squares:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(shape_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
for cnt in rectangles:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(shape_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
for cnt in triangles:
    cv2.drawContours(shape_img, [cnt], 0, (0, 255, 255), 2)
for cnt in hexagons:
    cv2.drawContours(shape_img, [cnt], 0, (255, 255, 0), 2)

# Save the shape image
#plt.imshow(shape_img)
print('Number of circles detected:', len(circles))
print('Number of rectangles detected:', len(rectangles))
print('Number of squares detected:', len(squares))
print('Number of triangles detected:', len(triangles))
print('Number of hexagons detected:', len(hexagons))


# In[ ]:




