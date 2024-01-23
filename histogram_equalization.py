from math import ceil, floor
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

###############################################
##      QUESTION 1     ##
###############################################

inp = cv.imread('test1.jpg', cv.IMREAD_GRAYSCALE)

#OPENCV HISTOGRAM EQUALIZATION
equ = cv.equalizeHist(inp)
cv.imwrite('output_2.png',equ)

#MY HISTOGRAM EQUALIZATION
hist,bins = np.histogram(inp.flatten(),256,[0,256])
hist = hist / inp.size
hist = hist * 255
cdf = hist.cumsum()
new = cdf[inp.flatten()]
my_out = new.reshape(inp.shape)
plt.imsave('output_1.png', my_out, cmap='gray')

#DIFFERENCE BETWEEN OPENCV AND MY HISTOGRAM EQUALIZATION

diff = abs(my_out-equ)
plt.imsave('diff.png', diff, cmap='gray')
total_diff = np.sum(diff)

print("Total difference for first method: " + total_diff.astype(str))



###############################################
##     QUESTION 2    ##
###############################################


inp = cv.imread('test1.jpg', cv.IMREAD_GRAYSCALE)

#Step 1, form histogram

H = np.histogram(inp.flatten(),256,[0,256])[0]

g_min = 0
for i in range(256):
    if H[i] > 0:
        g_min = i
        break

#Step 2, form cumulative histogram

H_c = np.cumsum(H)

H_min = H_c[g_min]

#Step 3, form transfer function

T = (( H_c - H_min ) / (inp.size - H_min))*255

Q = np.ceil(T - 0.5)
Q = Q.astype('uint8')

#Step 4, form output image

out = Q[inp.flatten()]
out = out.reshape(inp.shape)

plt.imsave('output_3.jpg', out, cmap='gray')

equ = cv.equalizeHist(inp)

diff = abs(out-equ)
total_diff = np.sum(diff)
print("Total difference for second method: " + total_diff.astype(str))