import argparse
import os
import os.path as osp
import numpy as np
from PIL import Image, ImageOps
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

    if args.data in ["base", "base2", "base3", "base7", "base5", "lighthouse"]:
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

        I = Image_Remap(I, 0, 1, -1, 1)
        ## Display
        fig, ax = plt.subplots()
        ax.imshow(I, cmap='gray', origin='lower')
        ax.set_title('Calibration Plate')
        if args.display:
            plt.show()

    return I


def load_kernel(args, ker_num, ker_type, Im_size):
    """
    Output:
        kernel: the 2D kernel matrix.
    """

    print("Use " + args.kernel + " kernel")
    N = int(np.ceil(0.05*Im_size))
    if N % 2 == 0:
        N = N + 1
    mid = int(np.ceil(N/2))
    fif = int(np.ceil(N/5))

    if ker_type == 0:
        # white: 1, black: 0
        kernel = np.ones((N, N)) * -1
        # red: bottom left of white square
        if ker_num == 0:
            kernel[0:mid, 0:mid] = 1
        # green: top right of white square
        elif ker_num == 1:
            kernel[mid-1:N+1, mid-1:N+1] = 1
        # blue: bottom left of white square
        elif ker_num == 2:
            kernel[0:mid, mid:N+1] = 1
        # yellow: top left of white square
        elif ker_num == 3:
            kernel[mid:N+1, 0:mid] = 1
    elif ker_type == 1:
        # white: 1, black: 0
        kernel = np.ones((N, N)) * -1
        # red:
        if ker_num == 0:
            kernel[0:mid, 0:mid] = 1
            kernel[mid - 1:N + 1, mid - 1:N + 1] = 1
        # green:
        elif ker_num == 1:
            kernel[0:mid, mid-1:N+1] = 1
            kernel[mid - 1:N + 1, 0:mid] = 1
        # blue:
        elif ker_num == 2:
            kernel[0:4, 0:4] = 1
            kernel[4:15, 4:15] = 1
        # yellow:
        elif ker_num == 3:
            kernel[mid-1:N+1, 0:mid] = 1
    elif ker_type == 2:
        # white: 1, black: 0
        kernel = np.ones((N, N))
        # red: bottom left of white square
        if ker_num == 0:
            kernel[0:mid, 0:mid] = 0
        # green: top right of white square
        elif ker_num == 1:
            kernel[mid-1:N+1, mid-1:N+1] = 0
        # blue: bottom left of white square
        elif ker_num == 2:
            kernel[0:mid, mid:N+1] = 0
        # yellow: top left of white square
        elif ker_num == 3:
            kernel[mid:N+1, 0:mid] = 0
    elif ker_type == 3:
        # white: 1, black: 0
        kernel = np.ones((N, N)) * -1
        # red:
        if ker_num == 0:
            kernel[0:mid, 0:mid] = 1
            kernel[mid - 1:N + 1, mid - 1:N + 1] = 1
        # green:
        elif ker_num == 1:
            kernel[0:mid, mid - 1:N + 1] = 1
            kernel[mid - 1:N + 1, 0:mid] = 1
        # blue:
        elif ker_num == 2:
            kernel[0:N-2, 2:N+1] = 1
        # yellow:
        elif ker_num == 3:
            kernel = np.ones((N, N))

    # kernel = kernel / np.sum(kernel)
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


def Image_Remap(I, old_min, old_max, new_min, new_max):
    for c in range(I.shape[2]):
        for n in range(I.shape[0]):
            for m in range(I.shape[1]):
                I[n, m, c] = (I[n, m, c] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    return I


def main(args):

    ## Load and display calibration grid and convolution kernels
    if int(args.current_step) >= 1:
        print("Load image")
        I = data_loader(args)
        corners = np.zeros(I.shape)
        for k in range(2,3):
            kernel = load_kernel(args, k, 3, I.shape[0])
            conv = Convolution(args, I, kernel)
            conv = conv / np.max(conv)
            for i in range(conv.shape[0]):
                for j in range(conv.shape[1]):
                    if np.average(conv[i, j, :]) >= 0.85:
                        if k == 0:
                            # pass
                            corners[i, j] = [1, 0, 0]
                            I[i, j] = [1, 0, 0]
                        elif k == 1:
                            corners[i, j] = [0, 1, 0]
                            I[i, j] = [0, 1, 0]
                        elif k == 2:
                            corners[i, j] = [0, 0, 1]
                            I[i, j] = [0, 0, 1]
                        elif k == 3:
                            pass
                            # corners[i, j] = [1, 1, 0]
                            # I[i, j] = [1, 1, 0]
                    # else:
                    #     corners[i, j] = [0, 0, 0]

            fig, ax = plt.subplots()
            ax.imshow(corners, cmap='gray', origin='lower')
            ax.set_title('Corner detection')
            if args.display:
                plt.show()
        ## Display
        # fig, ax = plt.subplots()
        # ax.imshow(conv, cmap='gray', origin='lower')
        # ax.set_title('Convolution')
        # if args.display:
        #     plt.show()



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
