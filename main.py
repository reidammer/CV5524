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
import torch.optim as optim
import matplotlib as mpl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Parameter
import json
import numpy as np
import os
import random
import time
import math
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
from scipy.optimize import curve_fit


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
        "test_case_3",
        "test_case_4",
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

    if args.curve in ["line1", "line2", "line3", "test_case_2", "test_case_1","cubic",
                      "exponential","quadratic","exponential","test_case_3","sinusoidal"]:
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


def train_model(model, train_loader, val_loader, num_epochs=25, use_curve_loss=True):
    """Train the model and monitor validation performance"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # run images through model
            outputs = model(images)

            # calculate loss
            batch_loss = F.mse_loss(outputs, labels)
            correct += torch.sum(torch.abs(outputs - labels) < 0.3).item()
            total += labels.numel()
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        train_accuracy = (correct / total) * 100
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.2f}%")

        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        # Validation, turn off gradients
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                # MSE loss between curve coefficients
                batch_loss = F.mse_loss(outputs, labels)

                running_val_loss += batch_loss.item()
                val_correct += torch.sum(torch.abs(outputs - labels) < 0.3).item()
                val_total += labels.numel()
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_accuracy = (val_correct / val_total) * 100
        val_accuracies.append(val_accuracy)
        scheduler.step(epoch_val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, ",
            f"Validation accuracy: {val_accuracy:.2f}%, ",
            f"Val Loss: {epoch_val_loss:.4f}",
        )

    return model, train_losses, val_losses


def test_model(model, test_loader, use_curve_loss=True):
    """Evaluate the model on the test set"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    running_loss = 0.0
    total_correct = 0
    total_predictions = 0
    # Test model, turn off gradients
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # MSE loss between curve coefficients
            batch_loss = F.mse_loss(outputs, labels)

            running_loss += batch_loss.item()
            total_correct += torch.sum(torch.abs(outputs - labels) < 0.3).item()
            total_predictions += labels.numel()
    test_accuracy = (total_correct / total_predictions) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    test_loss = running_loss / len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    return test_loss


def plot_curve_from_coeffs(ax, coeffs, x_range=(-10, 10), num_points=100):
    """Plot a curve on the given axis based on coefficients and curve type"""
    x = np.linspace(x_range[0], x_range[1], num_points)

    # Only linear curve for now
    y = coeffs[0] * x + coeffs[1]

    ax.plot(x, y)
    return x, y


def visualize_predictions(model, test_loader, num_examples=5):
    """Visualize model predictions against ground truth for some test examples"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    fig, axes = plt.subplots(num_examples, 2, figsize=(12, 4 * num_examples))

    examples_seen = 0
    # Test examples
    with torch.no_grad():
        for images, labels in test_loader:
            if examples_seen >= num_examples:
                break

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            for i in range(min(len(images), num_examples - examples_seen)):
                ax1 = axes[examples_seen, 1]

                # Plot the predicted and ground truth curves
                true_coeffs = labels[i].cpu().numpy()
                pred_coeffs = outputs[i].cpu().numpy()

                # Get coefficients for curves
                true_coeffs = true_coeffs[:2]
                pred_coeffs = pred_coeffs[:2]

                # Plot curves to compare
                plot_curve_from_coeffs(ax1, true_coeffs, x_range=(-5, 5))
                plot_curve_from_coeffs(ax1, pred_coeffs, x_range=(-5, 5))
                ax1.legend(["Ground Truth", "Prediction"])
                ax1.set_title(f"Curve Fitting Results")

                examples_seen += 1

    plt.tight_layout()
    plt.show()


class CurveCoefficientCNN(nn.Module):
    def __init__(self, num_coefficients=2):
        super(CurveCoefficientCNN, self).__init__()

        # Use a pre-trained CNN as the backbone
        self.backbone = models.resnet18(pretrained=True)

        # Replace the final FC layer with one for coefficient regression
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_coefficients)

    def forward(self, x):
        return self.backbone(x)


def generate_curve(x, coeffs):
    return x * coeffs[0] + coeffs[1]  # Example for linear curve

def sinusoidal(x, A, B, C, D):
    return A * np.sin(B*x + C) + D


def main(args):

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

    with open("./curve_projection_dataset/metadata.json", "r") as f:
        metadata = json.load(f)

    # Create dataset
    dataset = CurveDataset(metadata, "./curve_projection_dataset")

    # Split dataset 70 15 15
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    curve = curve_loader(args)
    if args.approach.lower() == 'advanced':

        print('Using Advanced Approach')

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        if int(args.current_step) >= 3 or int(args.current_step) == 0:
            # Initialize model
            model = CurveCoefficientCNN(num_coefficients=2)

            # Train model
            trained_model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, num_epochs=20, use_curve_loss=True
            )

            # Plot training and validation loss curves
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label="Training Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.show()

            # Test model
            test_loss = test_model(trained_model, test_loader, use_curve_loss=True)

            # Visualize results
            visualize_predictions(trained_model, test_loader, num_examples=5)

            # Save the model
            torch.save(trained_model.state_dict(), "curve_fitting_model.pth")
            print("Model saved as curve_fitting_model.pth")
            print("Model test loss: ", test_loss)

    else:
        if int(args.current_step) >= 3 or int(args.current_step) == 0:
            # Add names of any more lines to fit.
            linear_lines = ['line1', 'line2', 'test_case_1', 'test_case_2']
            quadratic_lines = ['quadratic', 'exponential']
            cubic_lines = ['cubic']
            sin_lines = ['sinusoidal', 'test_case_3']
            x = None
            y = None
            # linear least squares regression
            A = []
            b = []
            x_data = []
            for i in range(curve.shape[0]):
                for j in range(curve.shape[1]):
                    if curve[i, j, 1] == 0:
                        A.append([j, 1])
                        x_data.append(j)
                        b.append(i)
            A = np.array(A)
            b = np.array(b)
            x_data = np.array(x_data)

            if args.curve.lower() in linear_lines:
                alpha = np.linalg.lstsq(A, b, rcond=None)[0]
                print(alpha)
                x = np.linspace(0, curve.shape[1], curve.shape[1])
                y = alpha[0] * x + alpha[1]

            elif args.curve.lower() in quadratic_lines:
                p = np.polyfit(x_data, b, 2)
                x = np.linspace(0, curve.shape[1], curve.shape[1])
                y = p[0] * x**2 + p[1] * x + p[2]

            elif args.curve.lower() in cubic_lines:
                p = np.polyfit(x_data, b, 3)
                x = np.linspace(0, curve.shape[1], curve.shape[1])
                y = p[0] * x**3 + p[1] * x**2 + p[2]*x + p[3]

            elif args.curve.lower() in sin_lines:
                amp = (np.max(b) - np.min(b)) / 2
                per = x_data[-1] - x_data[0]
                initial = [amp,per,0,0]

                fit, cov = curve_fit(sinusoidal, x_data, b, initial)
                x = np.linspace(0, curve.shape[1], curve.shape[1])
                y = sinusoidal(x, fit[0], fit[1], fit[2], fit[3])


            fig, ax = plt.subplots()
            plt.plot(x, y)
            ax.imshow(curve, cmap='gray', origin='lower')
            ax.set_title('Calculated Curve Fit Over Captured Curve')
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
        ax.imshow(I, cmap='gray', origin='lower')
        ax.set_title('Corner Detection for Basis Calculation')
        if args.display:
            plt.show()


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
    parser.add_argument("--approach", default = "advanced", type = str)
    args = parser.parse_args()
    main(args)
