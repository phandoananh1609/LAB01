import argparse
import numpy as np
import cv2

def change_contrast(img, alpha):
    row, column, channel = img.shape
    img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    img=  img.astype(np.uint16) * alpha
    img = np.clip(img,0,255)
    img = img.reshape((row, column, channel))
    return img

def adjust_contrast(input_path, output_path, alpha):
    # Load input image
    img = cv2.imread(input_path)

    # Apply contrast adjustment
    img = change_contrast(img, alpha)

    # Save output image
    cv2.imwrite(output_path, img)

if __name__ == '__main__':
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Adjust contrast of an image')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image file path')
    parser.add_argument('--contrast', type=float, required=True,
                        help='Contrast adjustment factor (0-1)')

    # Parse command line arguments
    args = parser.parse_args()

    # Call contrast adjustment function
    adjust_contrast(args.input, args.output, args.contrast)