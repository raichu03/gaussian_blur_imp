from convolution import conv
import numpy as np
import cv2




obj = conv()

image = cv2.imread('image2.jpg')

img = cv2.resize(image, (500, 500))
img = img/255

kernel_size = 7
kernal = obj.gaussian_filter(kernel_size)


final_image = obj.blur_image(img, kernal)

cv2.imshow('original', img)
cv2.imshow('final', final_image)
cv2.waitKey(0)

cv2.destroyAllWindows()

