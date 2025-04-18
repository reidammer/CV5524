import argparse
import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


def data_loader(args):
    """
    Output:
        I: the image matrix of size N-by-M-by-3.
            The pixel values are within [0, 1].
            The first dimension is horizontal, left-right (N pixels).
            The second dimension is vertical, bottom-up (M pixels).
            The third dimension is color channels, from Red to Green to Blue.
    """

    if args.data in ["base2", "lighthouse"]:
        print("Using " + args.data + " photo")
        current_dir = os.getcwd()
        image_path = osp.join(current_dir, 'Calibration', args.data + '.png')
        I = np.asarray(Image.open(image_path)).astype(np.float64)/255
        I = np.transpose(I, (1, 0, 2))
        I = I[:, ::-1, 0:3]
        I = np.transpose(I, (1, 0, 2))

        ## Display
        fig, ax = plt.subplots()
        ax.imshow(I, cmap='gray', origin='lower')
        ax.set_title('Calibration Plate')
        if args.display:
            plt.show()

    return I


def load_kernel(args, ker_num):
    """
    Output:
        kernel: the 2D kernel matrix.
    """

    print("Use " + args.kernel + " kernel")

    # find bottom left corner of black squares
    # if args.kernel == "bot_left":
    if ker_num == 0:
        # white: 1, black: 0
        kernel = np.ones((11, 11))
        kernel[0:6, 0:6] = 0
    # find bottom left corner of white squares
    elif ker_num == 1:
        kernel = np.zeros((11, 11))
        kernel[0:6, 0:6] = 1
    # find top right corner of white squares
    elif ker_num == 2:
        kernel = np.zeros((11, 11))
        kernel[6:11, 6:11] = 1

        ## Display
        fig, ax = plt.subplots()
        ax.imshow(kernel, cmap='gray', origin='lower')
        ax.set_title('Convolution Kernel')
        if args.display:
            plt.show()

    return kernel


def Convolution(args, I, kernel):
    """
    Input:
        I: input image, a 3D matrix of size N-by-M-by-3
        kernel: convolutional kernel, a 2D matrix of size (2R + 1)-by-(2R + 1)
    Output:
        I_out: convolved image, a 3D matrix of size N-by-M-by-3
    """

    ## Initiate the output
    if np.size(I.shape) != 3:
        I = I.reshape((I.shape[0], I.shape[1], 1))
    I_out = np.zeros(I.shape)

    ## Zero padding
    R = int((kernel.shape[0]-1)/2)
    I_pad = np.concatenate((np.zeros((R, I.shape[1], 3)), I, np.zeros((R, I.shape[1], 3))), axis=0)
    I_pad = np.concatenate((np.zeros((I_pad.shape[0], R, 3)), I_pad, np.zeros((I_pad.shape[0], R, 3))), axis=1)

    ## Convolution
    for c in range(I_pad.shape[2]):
        for n in range(I_out.shape[0]):
            for m in range(I_out.shape[1]):
                I_out[n, m, c] = np.sum(I_pad[n : (n + kernel.shape[0]), m : (m + kernel.shape[1]), c] * kernel[::-1, ::-1])

    return I_out





def main(args):

    ## Load and display calibration grid and convolution kernels
    if int(args.current_step) >= 1:
        print("Load image")
        I = data_loader(args)
        corners = data_loader(args)
        for k in range(3):
            kernel = load_kernel(args, k)
            conv = Convolution(args, I, kernel)
            conv = conv / np.max(conv)
            for i in range(conv.shape[0]):
                for j in range(conv.shape[1]):
                    if np.average(conv[i, j, :]) >= 0.84:
                        if k == 0:
                            corners[i, j] = [1, 0, 0]
                            I[i, j] = [1, 0, 0]
                        elif k == 1:
                            corners[i, j] = [0, 1, 0]
                            I[i, j] = [0, 1, 0]
                        else:
                            corners[i, j] = [0, 0, 1]
                            I[i, j] = [0, 0, 1]
                    else:
                        corners[i, j] = [0, 0, 0]

        ## Display
        # fig, ax = plt.subplots()
        # ax.imshow(conv, cmap='gray', origin='lower')
        # ax.set_title('Convolution')
        # if args.display:
        #     plt.show()

        fig, ax = plt.subplots()
        ax.imshow(corners, cmap='gray', origin='lower')
        ax.set_title('Corner detection')
        if args.display:
            plt.show()

        fig, ax = plt.subplots()
        ax.imshow(I, cmap='gray', origin='lower')
        ax.set_title('Calibration Plate with Corner Detection')
        if args.display:
            plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convolution_and_Fourier_and_Filter")
    parser.add_argument('--path', default="data", type=str)
    parser.add_argument('--data', default="none", type=str)
    parser.add_argument('--kernel', default="binomial", type=str)
    parser.add_argument('--scale', default=3, type=int)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--current_step', default=1, type=int)
    args = parser.parse_args()
    main(args)
