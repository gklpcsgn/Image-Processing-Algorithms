import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt

# KODUN CALISMASI BENIM BILGISAYARIMDA 15DK SURUYOR, OUTPUT OLARAK KAYDETTIM SUANKI PARAMETRELERLE YINE DE ISTERSENIZ CALISTIRABILIRSINIZ


gaussian_01 = cv.imread('noisyImage_Gaussian_01.jpg',0)
gaussian = cv.imread('noisyImage_Gaussian.jpg',0)
ground_truth = cv.imread('lena_grayscale_hq.jpg',0)

def my_non_local_means_denoising(image, h, templateWindowSize, searchWindowSize):
    #For each pixel p of the input (noisy) image:
    # 1. Define a local, small, fixed size (5x5, 7x7) patch centered at p.
    # 2. Construct a vector V p from the pixel values in this patch (flatten).
    # 3. To compute the output value of pixel p:
        # i. Raster-scan the search window centered at p.
        # ii. Compute the similarity weight as a function of |V p – V q |.
        # iii. Then compute the weighted average of all pixels q.

    output = np.zeros(image.shape)
    padding = int(templateWindowSize/2)
    img = cv.copyMakeBorder(image, padding, padding, padding, padding, cv.BORDER_REFLECT)
    
    temp = h**2

    for i in range(padding, img.shape[0]-padding):
        for j in range(padding, img.shape[1]-padding):
            patch = img[i-padding:i+padding+1, j-padding:j+padding+1].flatten().astype(np.float64)
            Z = 0
            if i - searchWindowSize // 2 < 0:
                start_i = 0
            else:
                start_i = i - searchWindowSize // 2
            
            if i + searchWindowSize // 2 + templateWindowSize >= img.shape[0]:
                end_i = img.shape[0] - templateWindowSize
            else:
                end_i = i + searchWindowSize // 2
            
            if j - searchWindowSize // 2 < 0:
                start_j = 0
            else:
                start_j = j - searchWindowSize // 2
            
            if j + searchWindowSize // 2 + templateWindowSize >= img.shape[1]:
                end_j = img.shape[1] - templateWindowSize
            else:
                end_j = j + searchWindowSize // 2

            weights = np.array([])
            for k in range(start_i, end_i):
                for l in range(start_j, end_j):
                    patch2 = img[k:k+templateWindowSize , l:l+templateWindowSize].flatten().astype(np.float64)
                    distance = np.sqrt(np.sum(np.square(patch - patch2))) / temp
                    exp = np.sum(np.exp(-distance))
                    weights = np.append(weights, exp)
                    Z += exp
            value = np.sum(weights * img[ start_i : end_i , start_j  : end_j ].flatten()) / Z
            output[i-padding, j-padding] = value
    return output

# filter strength for gaussian
h = 1
templateWindowSize = 7
searchWindowSize = 15

# filter strength for gaussian_01
h_01 = 1
templateWindowSize_01 = 7
searchWindowSize_01 = 15


cv_nlm = cv.fastNlMeansDenoising(gaussian,None,h,templateWindowSize,searchWindowSize)
cv_gaussian = cv.GaussianBlur(gaussian,(5,5),0)
my_nlm = my_non_local_means_denoising(gaussian, h, templateWindowSize, searchWindowSize)

cv_nlm_01 = cv.fastNlMeansDenoising(gaussian_01,None,h_01,templateWindowSize_01,searchWindowSize_01)
cv_gaussian_01 = cv.GaussianBlur(gaussian_01,(5,5),0)
my_nlm_01 = my_non_local_means_denoising(gaussian_01, h_01, templateWindowSize_01, searchWindowSize_01)

# plot the images
plt.figure(figsize=(50,60),dpi=150)

plt.subplot(2,3,1),plt.imshow(my_nlm_01,cmap = 'gray')
plt.title('My NLM gaussian_01'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2),plt.imshow(cv_nlm_01,cmap = 'gray')
plt.title('NLM Denoising gaussian_01'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,3),plt.imshow(cv_gaussian_01,cmap = 'gray')
plt.title('Gaussian Blur gaussian_01'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,4),plt.imshow(my_nlm,cmap = 'gray')
plt.title('My NLM gaussian'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,5),plt.imshow(cv_nlm,cmap = 'gray')
plt.title('NLM Denoising gaussian'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,6),plt.imshow(cv_gaussian,cmap = 'gray')
plt.title('Gaussian Blur gaussian'), plt.xticks([]), plt.yticks([])

plt.show()

