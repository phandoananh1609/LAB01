import argparse
import numpy as np
import cv2
import math

def convolution(image, kernel, average=False):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    result = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image


    for row in range(image_row):
        for col in range(image_col):
            result[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                result[row, col] /= kernel.shape[0] * kernel.shape[1]

    
    
    return result

def calc_h(x, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power(x / sd, 2) / 2)

def gaussian_kernel(size, sigma):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = calc_h(kernel_1D[i], sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()


    return kernel_2D
def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
    return convolution(image, kernel, average=True)

def gaussian_filter(img_path, output_path, kernel_size):
    # Load image
    img = cv2.imread(img_path)

    # Apply Gaussian filter
    filtered_img = gaussian_blur(img, 3)

    # Save output image
    cv2.imwrite(output_path, filtered_img)

def median_kernel(kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return kernel

def median_blur(image, kernel_size):
    kernel = median_kernel(kernel_size)
    return convolution(image, kernel)

def median_filter(img_path, output_path, kernel_size):
    # Load image
    img = cv2.imread(img_path)

    # Apply median filter
    filtered_img = median_blur(img, kernel_size)

    # Save output image
    cv2.imwrite(output_path, filtered_img)

def average_kernel(size):
    kernel = np.ones((size, size), np.float32) / (size*size)
    return kernel

def average_filter(img_path, output_path, kernel_size):
    # Load image
    img = cv2.imread(img_path)

    # Define kernel
    kernel = average_kernel(kernel_size)

    # Apply filter
    filtered_img = convolution(img, kernel, average=False)

    # Save output image
    cv2.imwrite(output_path, filtered_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply image filters')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image file path')
    parser.add_argument('--filter', type=str, required=True, choices=['gau', 'med', 'avg'],
                        help='Type of filter to apply (gau, med, or avg)')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Kernel size for the filter (odd number)')

    args = parser.parse_args()

    if args.filter == 'gau':
        gaussian_filter(args.input, args.output, args.kernel_size)
    elif args.filter == 'med':
        median_filter(args.input, args.output, args.kernel_size)
    elif args.filter == 'avg':
        average_filter(args.input, args.output, args.kernel_size)


        