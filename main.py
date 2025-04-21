import argparse
import json
import os
import os.path as osp
from matplotlib import transforms
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Parameter
import json
import numpy as np
import os
import random
import time
import math
from matplotlib.path import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch


def calibration_loader(args):
    """
    Output:
        I: the image matrix of size N-by-M-by-3.
            The pixel values are within [0, 1].
            The first dimension is horizontal, left-right (N pixels).
            The second dimension is vertical, bottom-up (M pixels).
            The third dimension is color channels, from Red to Green to Blue.
    """

    if args.calibration in [
        "base",
        "base2",
        "base3",
        "base7",
        "base5",
        "test_case_2",
        "test_case_1",
        "lighthouse",
    ]:
        print("Using " + args.calibration + " photo")
        current_dir = os.getcwd()
        image_path = osp.join(current_dir, "Calibration", args.calibration + ".png")
        I = np.asarray(Image.open(image_path)).astype(np.float64) / 255
        I = np.transpose(I, (1, 0, 2))
        I = I[:, ::-1, 0:3]
        I = np.transpose(I, (1, 0, 2))

        ## Display
        fig, ax = plt.subplots()
        ax.imshow(I, cmap="gray", origin="lower")
        ax.set_title("Calibration Plate")
        if args.display:
            plt.show()

    return I


def curve_loader(args):
    """
    Output:
        I: the image matrix of size N-by-M-by-3.
            The pixel values are within [0, 1].
            The first dimension is horizontal, left-right (N pixels).
            The second dimension is vertical, bottom-up (M pixels).
            The third dimension is color channels, from Red to Green to Blue.
    """

    if args.curve in ["line1", "line2", "line3", "test_case_2", "test_case_1"]:
        print("Using " + args.curve + " photo")
        current_dir = os.getcwd()
        image_path = osp.join(current_dir, "Curves", args.curve + ".png")
        I = np.asarray(Image.open(image_path)).astype(np.float64) / 255
        I = np.transpose(I, (1, 0, 2))
        I = I[:, ::-1, 0:3]
        I = np.transpose(I, (1, 0, 2))

        ## Display
        fig, ax = plt.subplots()
        ax.imshow(I, cmap="gray", origin="lower")
        ax.set_title("Captured Curve")
        if args.display:
            plt.show()

    return I


def load_kernel(args, ker_num, ker_type, Im_size):
    """
    Output:
        kernel: the 2D kernel matrix.
    """

    N = int(np.ceil(0.05 * Im_size))
    if N % 2 == 0:
        N = N + 1
    mid = int(np.ceil(N / 2))

    if ker_type == 0:
        # white: 1, black: 0
        kernel = np.ones((N, N)) * -1
        # red:
        if ker_num == 0:
            print("Use positive diagonal corner detection kernel")
            kernel[0:mid, 0:mid] = 1
            kernel[mid - 1 : N + 1, mid - 1 : N + 1] = 1
        # green:
        elif ker_num == 1:
            print("Use negative diagonal corner detection kernel")
            kernel[0:mid, mid - 1 : N + 1] = 1
            kernel[mid - 1 : N + 1, 0:mid] = 1
        # blue:
        elif ker_num == 2:
            print("Use origin detection kernel")
            m = int(np.floor(1 / 15 * N))
            kernel[0 : 5 * m, :] = 1
            kernel[7 * m : N, 0 : 8 * m] = 1
            kernel[9 * m : N, 10 * m : N] = 1

    ## Display
    fig, ax = plt.subplots()
    ax.imshow(kernel, cmap="gray", origin="lower")
    ax.set_title("Convolution Kernel")
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
    R = int((kernel.shape[0] - 1) / 2)
    I_pad = np.concatenate(
        (np.zeros((R, I.shape[1], 3)), I, np.zeros((R, I.shape[1], 3))), axis=0
    )
    I_pad = np.concatenate(
        (np.zeros((I_pad.shape[0], R, 3)), I_pad, np.zeros((I_pad.shape[0], R, 3))),
        axis=1,
    )

    ## Convolution
    for c in range(I_pad.shape[2]):
        for n in range(I_out.shape[0]):
            for m in range(I_out.shape[1]):
                I_out[n, m, c] = np.sum(
                    I_pad[n : (n + kernel.shape[0]), m : (m + kernel.shape[1]), c]
                    * kernel[::-1, ::-1]
                )

    return I_out


class CurveDataset(Dataset):
    def __init__(self, metadata, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.metadata = metadata["curves"]
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = self.root_dir / item["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        coeffs = item["coefficients"]
        label = torch.tensor(list(coeffs.values()), dtype=torch.float32)

        return image, label


def Image_Remap(I, old_min, old_max, new_min, new_max):
    for c in range(I.shape[2]):
        for n in range(I.shape[0]):
            for m in range(I.shape[1]):
                I[n, m, c] = (I[n, m, c] - old_min) / (old_max - old_min) * (
                    new_max - new_min
                ) + new_min

    return I


class CurveFittingNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64, 32], output_dim=1):
        super(CurveFittingNetwork, self).__init__()

        layers = []
        prev_dim = input_dim

        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def main(args):

    # with open("./curve_projection_dataset/metadata.json", "r") as f:
    #     metadata = json.load(f)

    # curve_data = metadata["curves"]

    # dataset = CurveDataset(metadata, "./curve_projection_dataset")
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, test_size]
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=32, shuffle=True
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=32, shuffle=False
    # )
    ## Load and display calibration grid and convolution kernels
    ## Detect corners and find origin
    if int(args.current_step) >= 1:
        print("Load image")
        I = calibration_loader(args)
        corners = np.zeros(I.shape)
        for k in range(3):
            kernel = load_kernel(args, k, 0, I.shape[0])
            conv = Convolution(args, I, kernel)
            conv = conv / np.max(conv)
            for i in range(conv.shape[0]):
                for j in range(conv.shape[1]):
                    if np.average(conv[i, j, :]) >= args.detection_threshold:
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
                            corners[i, j] = [1, 0, 1]
                            I[i, j] = [1, 0, 1]
                    # else:
                    #     corners[i, j] = [0, 0, 0]

            fig, ax = plt.subplots()
            ax.imshow(corners, cmap="gray", origin="lower")
            ax.set_title("Corner detection")
            if args.display:
                plt.show()

        fig, ax = plt.subplots()
        ax.imshow(I, cmap="gray", origin="lower")
        ax.set_title("Calibration Plate with Corner Detection")
        if args.display:
            plt.show()

    ## Define coordinate system
    if int(args.current_step) >= 2:
        N = I.shape[0]
        M = I.shape[1]
        off = 5

        # green detects starting point
        x1 = 0
        y1 = 0
        detected = False
        for n in range(N):
            for m in range(M):
                if (
                    I[n, m, 0] == 0
                    and I[n, m, 1] == 1
                    and I[n, m, 2] == 0
                    and not detected
                ):
                    I[n, m] = [1, 0, 1]
                    detected = True
                    x1 = n
                    y1 = m

        # red detects ending point
        x2 = x1
        y2 = y1
        xdist = I.shape[0]
        for n in range(N):
            for m in range(y1 + off, M):
                if I[n, m, 0] == 1 and I[n, m, 1] == 0 and I[n, m, 2] == 0:
                    if np.sqrt(pow((n - x1), 2) + pow((m - y1), 2)) < xdist:
                        x2 = n
                        y2 = m
                        xdist = np.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))
        I[x2, y2] = [0, 1, 1]
        x3 = x1
        y3 = y1
        ydist = I.shape[1]
        for m in range(M):
            for n in range(x1 + off, N):
                if I[n, m, 0] == 1 and I[n, m, 1] == 0 and I[n, m, 2] == 0:
                    if np.sqrt(pow((n - x1), 2) + pow((m - y1), 2)) < ydist:
                        x3 = n
                        y3 = m
                        ydist = np.sqrt(pow((x1 - x3), 2) + pow((y1 - y3), 2))
        I[x3, y3] = [0, 1, 1]

        # blue detects origin
        xo = 0
        yo = 0
        detected = False
        for n in range(N):
            for m in range(M):
                if (
                    I[n, m, 0] == 0
                    and I[n, m, 1] == 0
                    and I[n, m, 2] == 1
                    and not detected
                ):
                    I[n, m] = [1, 1, 0]
                    detected = True
                    xo = n
                    yo = m

        fig, ax = plt.subplots()
        ax.imshow(I, cmap="gray", origin="lower")
        ax.set_title("Corner Detection for Basis Calculation")
        if args.display:
            plt.show()

    ## Curve fitting
    ## THIS IS WHERE NEURAL NETWORK WOULD BE

    curves_dir = os.path.join("curve_projection_dataset", "curves")

    if int(args.current_step) >= 3 or int(args.current_step) == 0:

        for curve_file in os.listdir(curves_dir):
            if curve_file.endswith(".png"):  # Process only PNG files
                curve_path = os.path.join(curves_dir, curve_file)
                print(f"Processing curve: {curve_file}")

                # Load the curve image
                curve = np.asarray(Image.open(curve_path)).astype(np.float64) / 255
                curve = np.transpose(curve, (1, 0, 2))
                curve = curve[:, ::-1, 0:3]
                curve = np.transpose(curve, (1, 0, 2))

            # linear least squares regression
            A = []
            b = []
            for i in range(curve.shape[0]):
                for j in range(curve.shape[1]):
                    if (
                        curve[i, j, 0] == 1
                        and curve[i, j, 1] == 0
                        and curve[i, j, 2] == 0
                    ):  # Red curve
                        A.append([j, 1])
                        b.append(i)
            A = np.array(A)
            b = np.array(b)
            alpha = np.linalg.lstsq(A, b, rcond=None)[0]
            print(alpha)

            x = np.linspace(0, curve.shape[1], curve.shape[1])
            y = alpha[0] * x + alpha[1]
            fig, ax = plt.subplots()
            plt.plot(x, y)
            ax.imshow(curve, cmap="gray", origin="lower")
            ax.set_title("Calculated Curve Fit Over Captured Curve")
            if args.display:
                plt.show()

    ## Projection
    if int(args.current_step) >= 4:
        x = []
        y = []
        for i in range(curve.shape[0]):
            for j in range(curve.shape[1]):
                if curve[i, j, 1] == 0:
                    x.append((i - xo) * 10 / xdist)
                    y.append((j - yo) * 10 / ydist)
                    I[i, j] = [1, 0, 1]
        x = np.array(x)
        y = np.array(y)
        print(x)
        print(y)
        fig, ax = plt.subplots()
        ax.imshow(I, cmap="gray", origin="lower")
        ax.set_title("Corner Detection for Basis Calculation")
        if args.display:
            plt.show()

        k = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convolution_and_Fourier_and_Filter")
    parser.add_argument("--path", default="data", type=str)
    parser.add_argument("--calibration", default="none", type=str)
    parser.add_argument("--curve", default="none", type=str)
    parser.add_argument("--kernel", default="binomial", type=str)
    parser.add_argument("--scale", default=3, type=int)
    parser.add_argument("--detection_threshold", default=0.95, type=float)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--current_step", default=1, type=int)
    args = parser.parse_args()
    main(args)
