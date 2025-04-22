import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
import random
from scipy import interpolate


class CalibrationCurveDataGenerator:
    def __init__(self, output_dir="./curve_projection_data"):
        """Initialize the data generator for calibration and curve projection."""
        self.output_dir = output_dir
        self.calibration_dir = os.path.join(output_dir, "calibration")
        self.curves_dir = os.path.join(output_dir, "curves")
        self.metadata_file = os.path.join(output_dir, "metadata.json")
        self.metadata = {"calibration": {}, "curves": []}

        # Create directories if they don't exist
        os.makedirs(self.calibration_dir, exist_ok=True)
        os.makedirs(self.curves_dir, exist_ok=True)

        # Calibration parameters
        self.square_size_mm = 10  # Each square is 10mm x 10mm
        self.board_size = (8, 8)  # 8x8 checkerboard
        self.calibration_complete = False
        self.pixels_per_mm = None
        self.origin_point = None
        self.checkerboard = None  # Store the calibration image

    def generate_calibration_plate(self):
        """Generate a blank calibration plate with a marker at the origin (0,0) in the center of the image."""
        board_h, board_w = self.board_size
        square_size_px = 50  # Use this to determine overall image scale

        # Image size in pixels
        height = board_h * square_size_px
        width = board_w * square_size_px
        plate = np.ones((height, width), dtype=np.uint8) * 255  # Blank white image

        # Define origin in pixel space as the image center
        center_x = width // 2
        center_y = height // 2

        # Draw a small cross marker at the center (origin)
        marker_len = square_size_px // 5
        cv2.line(
            plate,
            (center_x - marker_len, center_y),
            (center_x + marker_len, center_y),
            0,
            2,
        )
        cv2.line(
            plate,
            (center_x, center_y - marker_len),
            (center_x, center_y + marker_len),
            0,
            2,
        )

        # Save the image
        calib_image_path = os.path.join(self.calibration_dir, "calibration_plate.png")
        cv2.imwrite(calib_image_path, plate)

        # Store the plate for curve drawing later
        self.checkerboard = plate
        self.calibration_complete = True

        # Pixels per mm (same scale as before)
        self.pixels_per_mm = square_size_px / self.square_size_mm

        # Set origin at the center of the image
        self.origin_point = (center_x, center_y)

        self.metadata["calibration"] = {
            "image_path": os.path.relpath(calib_image_path, self.output_dir),
            "square_size_mm": self.square_size_mm,
            "board_size": self.board_size,
            "pixels_per_mm": self.pixels_per_mm,
            "origin_point": self.origin_point,
            "center_point": (center_x, center_y),
        }

        return calib_image_path

    def mm_to_pixel(self, x_mm, y_mm):
        """Convert real-world coordinates (mm) to pixel coordinates."""
        if not self.calibration_complete:
            raise ValueError("Calibration must be completed first")

        # Convert from mm to pixels, adjusting for origin
        x_px = self.origin_point[0] + x_mm * self.pixels_per_mm
        y_px = (
            self.origin_point[1] - y_mm * self.pixels_per_mm
        )  # y increases upward in real world

        return int(x_px), int(y_px)

    def pixel_to_mm(self, x_px, y_px):
        """Convert pixel coordinates to real-world coordinates (mm)."""
        if not self.calibration_complete:
            raise ValueError("Calibration must be completed first")

        # Convert from pixels to mm, adjusting for origin
        x_mm = (x_px - self.origin_point[0]) / self.pixels_per_mm
        y_mm = (
            self.origin_point[1] - y_px
        ) / self.pixels_per_mm  # y increases upward in real world

        return x_mm, y_mm

    def generate_curve_function(self, x_range_mm, y_range_mm):
        """Generate a random curve function and its equation within the visible mm range."""
        curve_types = ["linear"]
        curve_type = random.choice(curve_types)

        x_min_mm, x_max_mm = x_range_mm
        x_span = x_max_mm - x_min_mm

        if curve_type == "linear":
            a = random.uniform(-0.5, 0.5)  # gentler slope
            b = random.uniform(
                y_range_mm[0] + 0.1 * x_span, y_range_mm[1] - 0.1 * x_span
            )

            equation = {
                "type": "linear",
                "form": f"y = {a:.4f}x + {b:.4f}",
                "coefficients": {"a": a, "b": b},
                "function": lambda x: a * x + b,
            }

        elif curve_type == "quadratic":
            a = random.uniform(-0.01, 0.01)
            b = random.uniform(-0.5, 0.5)
            c = random.uniform(*y_range_mm)

            equation = {
                "type": "quadratic",
                "form": f"y = {a:.4f}x² + {b:.4f}x + {c:.4f}",
                "coefficients": {"a": a, "b": b, "c": c},
                "function": lambda x: a * x**2 + b * x + c,
            }

        elif curve_type == "cubic":
            a = random.uniform(-0.0002, 0.0002)
            b = random.uniform(-0.01, 0.01)
            c = random.uniform(-0.5, 0.5)
            d = random.uniform(*y_range_mm)

            equation = {
                "type": "cubic",
                "form": f"y = {a:.6f}x³ + {b:.4f}x² + {c:.4f}x + {d:.4f}",
                "coefficients": {"a": a, "b": b, "c": c, "d": d},
                "function": lambda x: a * x**3 + b * x**2 + c * x + d,
            }

        elif curve_type == "exponential":
            a = random.uniform(1, 5)
            b = random.uniform(0.01, 0.03)
            c = random.uniform(*y_range_mm)

            equation = {
                "type": "exponential",
                "form": f"y = {a:.4f} * e^({b:.4f}x) + {c:.4f}",
                "coefficients": {"a": a, "b": b, "c": c},
                "function": lambda x: a * np.exp(b * x) + c,
            }

        elif curve_type == "sinusoidal":
            a = random.uniform(5, 10)
            b = random.uniform(0.05, 0.1)
            c = random.uniform(0, 2 * np.pi)
            d = random.uniform(*y_range_mm)

            equation = {
                "type": "sinusoidal",
                "form": f"y = {a:.4f} * sin({b:.4f}x + {c:.4f}) + {d:.4f}",
                "coefficients": {"a": a, "b": b, "c": c, "d": d},
                "function": lambda x: a * np.sin(b * x + c) + d,
            }

        return equation, x_min_mm, x_max_mm

    def draw_curve(self, equation, index, x_min_mm, x_max_mm):
        """Generate an image with the curve drawn on top of the calibration plate."""
        if not self.calibration_complete or self.checkerboard is None:
            raise ValueError("Calibration must be completed first")

        # Create a copy of the calibration image
        # Convert to BGR for colored curve
        img = cv2.cvtColor(self.checkerboard.copy(), cv2.COLOR_GRAY2BGR)

        # Define image size
        height, width = img.shape[:2]

        # Generate points along the curve in mm
        x_mm = np.linspace(x_min_mm, x_max_mm, 1000)
        y_mm = equation["function"](x_mm)

        # Convert to pixel coordinates
        curve_pixels = [self.mm_to_pixel(x, y) for x, y in zip(x_mm, y_mm)]
        curve_pixels = np.array(curve_pixels)

        # Filter out points outside the image
        valid_points = (
            (curve_pixels[:, 0] >= 0)
            & (curve_pixels[:, 0] < width)
            & (curve_pixels[:, 1] >= 0)
            & (curve_pixels[:, 1] < height)
        )

        # Check if we have valid points
        if not np.any(valid_points):
            # No valid points - regenerate a different curve
            print(f"Curve {index} outside image bounds, regenerating...")
            new_equation, new_x_min, new_x_max = self.generate_curve_function()
            return self.draw_curve(new_equation, index, new_x_min, new_x_max)

        # Filter to valid points
        filtered_x_mm = x_mm[valid_points]
        filtered_y_mm = y_mm[valid_points]
        curve_pixels = curve_pixels[valid_points]

        # Draw the curve (thick RED line)
        red_color = (0, 0, 255)  # BGR format: red
        for i in range(len(curve_pixels) - 1):
            cv2.line(
                img, tuple(curve_pixels[i]), tuple(curve_pixels[i + 1]), red_color, 3
            )  # thicker line for better visibility

        # Add some randomness to simulate hand-drawing
        if len(curve_pixels) > 0:
            # Add some "hand-drawn" noise to the curve
            noise_points = []
            for i in range(len(curve_pixels) - 1):
                if i % 5 == 0:  # Add noise every few points
                    noise_x = curve_pixels[i][0] + random.randint(-3, 3)
                    noise_y = curve_pixels[i][1] + random.randint(-3, 3)
                    noise_points.append((noise_x, noise_y))

            # Draw the noise points
            for pt in noise_points:
                cv2.circle(img, pt, 1, red_color, -1)

        # Save the curve image
        curve_image_path = os.path.join(self.curves_dir, f"curve_{index}.png")
        cv2.imwrite(curve_image_path, img)

        # Store points for reference (a sample of points in mm)
        if len(filtered_x_mm) > 20:
            sample_indices = np.linspace(0, len(filtered_x_mm) - 1, 20, dtype=int)
        else:
            # If we have fewer than 20 points, use all of them
            sample_indices = np.arange(len(filtered_x_mm))

        sample_points_mm = [
            (float(filtered_x_mm[i]), float(filtered_y_mm[i])) for i in sample_indices
        ]

        # Store curve data
        curve_data = {
            "id": index,
            "image_path": os.path.relpath(curve_image_path, self.output_dir),
            "equation_type": equation["type"],
            "equation_form": equation["form"],
            "coefficients": equation["coefficients"],
            "sample_points_mm": sample_points_mm,
            "x_range_mm": [x_min_mm, x_max_mm],
        }

        self.metadata["curves"].append(curve_data)
        return curve_image_path

    def generate_dataset(self, num_curves=50):
        """Generate a dataset with curves drawn on top of the calibration plate."""
        print("Generating calibration plate...")
        self.generate_calibration_plate()

        # Compute visible area in mm
        height, width = self.checkerboard.shape[:2]
        w_mm = width / self.pixels_per_mm
        h_mm = height / self.pixels_per_mm

        x_range_mm = (-w_mm / 2 + 5, w_mm / 2 - 5)  # margin of 5mm
        y_range_mm = (-h_mm / 2 + 5, h_mm / 2 - 5)

        print(f"Visible mm range X: {x_range_mm}, Y: {y_range_mm}")

        print(f"Generating {num_curves} curve images...")
        for i in range(num_curves):
            equation, x_min_mm, x_max_mm = self.generate_curve_function(
                x_range_mm, y_range_mm
            )
            self.draw_curve(equation, i, x_min_mm, x_max_mm)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_curves} curves")

        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=4)

        print(f"Dataset generation complete. Saved to {self.output_dir}")


# Usage example
if __name__ == "__main__":
    generator = CalibrationCurveDataGenerator(output_dir="./curve_projection_dataset")
    generator.generate_dataset(num_curves=500)  # Generate 50 curve samples
