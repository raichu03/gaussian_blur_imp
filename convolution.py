import numpy as np
import cv2


class conv:

    def __init__(self):
        self.kernal_height =  7
        self.kernal_weidth = 7
        self.sigma = 50.0

    def matrix_multiply(self, i, j, image, kernal):
        """performs matrix multiplication of image and kernal"""
        sum = 0
        for m in range(self.kernal_height):
            for n in range(self.kernal_weidth):
                sum = sum + image[i+m][j+n] * kernal[m][n]
        return sum
    
    
    def gaussian_filter(self, kernel_size):
        """generates and returns gaussian filter matrix of given kernel size"""

        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*self.sigma**2)) * np.exp(-((x-(kernel_size-1)/2)**2 + (y-(kernel_size-1)/2)**2)/(2*self.sigma**2)),
            (kernel_size, kernel_size)
        )
        return kernel / np.sum(kernel)

        

    def convolution(self, image, kernal):
        """performs convolution of image and kernal and returns final image"""

        image_height, image_weidth = image.shape

        final_height = image_height - self.kernal_height + 1 
        final_weidth = image_weidth - self.kernal_weidth + 1

        final_image = np.zeros((final_height, final_weidth))

        for i in range(final_height):
            for j in range(final_weidth):
                final_image[i][j] = self.matrix_multiply(i, j, image, kernal)
        
        return final_image
    
    def blur_image(self, image, kernel):

        blue, green, red = cv2.split(image)

        blue_img = self.convolution(blue, kernel)
        green_img = self.convolution(green, kernel)
        red_img = self.convolution(red, kernel)

        final_image = cv2.merge((blue_img, green_img, red_img))

        return final_image



