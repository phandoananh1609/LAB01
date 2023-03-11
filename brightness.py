import argparse
import numpy as np
import cv2

def change_brightness(img, bias):
    row, column, channel = img.shape
    img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    img=  img.astype(np.uint16) + bias
    img = np.clip(img,0,255)
    img = img.reshape((row, column, channel))
    return img

def adjust_brightness(input_path, output_path, bias):
    # Load input image
    img = cv2.imread(input_path)

    # Apply brightness adjustment
    img = change_brightness(img, bias)

    # Save output image
    cv2.imwrite(output_path, img)

if __name__ == '__main__':
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Adjust brightness of an image')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image file path')
    parser.add_argument('--brightness', type=float, required=True,
                        help='Brightness adjustment factor (0-1)')

    # Parse command line arguments
    args = parser.parse_args()

    # Call brightness adjustment function
    adjust_brightness(args.input, args.output, args.brightness)