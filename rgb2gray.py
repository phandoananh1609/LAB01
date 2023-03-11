import numpy as np
import matplotlib.pyplot as plt
import argparse

def rgb2gray(rgb):
    """
    Chuyển đổi một hình ảnh RGB thành hình ảnh Grayscale.
    
    Tham số đầu vào:
    rgb (numpy.ndarray): Mảng 3 chiều chứa hình ảnh RGB.
    
    Trả về:
    numpy.ndarray: Mảng 2 chiều chứa hình ảnh Grayscale.
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)

def main(input_path, output_path):
    # Load input image
    img = plt.imread(input_path)

    # Convert to grayscale
    gray = rgb2gray(img)

    # Save output image
    plt.imsave(output_path, gray, cmap='gray')

    # Show input and output images side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img)
    axs[0].set_title('Input')
    axs[1].imshow(gray, cmap='gray')
    axs[1].set_title('Output')
    plt.show()

if __name__ == '__main__':
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Convert RGB image to grayscale')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image file path')

    # Parse command line arguments
    args = parser.parse_args()

    # Run main program
    main(args.input, args.output)