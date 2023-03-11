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

def gradient_magnitude(img_x, img_y):
    gradient_magnitude = np.sqrt(np.square(img_x) + np.square(img_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    return gradient_magnitude

    

def sobel_detect(img_path, output_path, kernel_size):
    # Load image
    img = cv2.imread(img_path)

    # Define kernel
    Wx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=int)
    Wx = Wx / 4
    Wy = np.array([[-1,-2,-1], [0,0,0],[1,2,1]], dtype=int)
    Wy = Wy / 4

    # Apply filter
    filtered_img = gradient_magnitude(convolution(img, Wx, average=True),convolution(img, Wy, average=True))

    # Save output image
    cv2.imwrite(output_path, filtered_img)   
    
def prewitt_detect(img_path, output_path, kernel_size):
    # Load image
    img = cv2.imread(img_path)

    # Define kernel
    Wx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=int)
    Wy = np.array([[-1,-1,-1], [0,0,0],[1,1,1]], dtype=int)
    Wx = Wx / 3
    Wy = Wy / 3

    # Apply filter
    filtered_img = gradient_magnitude(convolution(img, Wx, average=True),convolution(img, Wy, average=True))

    # Save output image
    cv2.imwrite(output_path, filtered_img)  

def laplace_detect(img_path, output_path, kernel_size):
    # Load image
    img = cv2.imread(img_path)

    # Define kernel
    W = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=int)

    # Apply filter
    filtered_img = convolution(img, W)

    # Save output image
    cv2.imwrite(output_path, filtered_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply image edge detect')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image file path')
    parser.add_argument('--edge', type=str, required=True, choices=['sobel', 'prewitt', 'laplace'],
                        help='Type of filter to apply (sobel filter, prewitt filter, or laplace filter)')
    parser.add_argument('--size', type=int, default=3,
                        help='Kernel size for the edge (odd number)')

    args = parser.parse_args()

    if args.edge == 'sobel':
        sobel_detect(args.input, args.output, args.size)
    elif args.edge == 'prewitt':
        prewitt_detect(args.input, args.output, args.size)
    elif args.edge == 'laplace':
        laplace_detect(args.input, args.output, args.size)