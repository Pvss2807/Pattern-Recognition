# Image Processing with Shape Detection and Deblurring

This project applies image deblurring and shape detection using OpenCV and NumPy. It detects geometric shapes (circles, rectangles, squares, triangles, hexagons) and visualizes the results.

---

## 📋 Features
- Deblurs images using Wiener filters.
- Detects geometric shapes (circles, rectangles, squares, triangles, hexagons).
- Visualizes detected edges and shapes on processed images.
- Uses OpenCV, NumPy, and Matplotlib.

---

## ⚙️ Setup Instructions
1. Clone or download this repository.
2. Install dependencies:
    ```bash
    pip install opencv-python numpy matplotlib
    ```
3. Place your images in a folder (update paths in the script if needed).

---

## 🚀 How to Run
1. Open the `image_processing.py` script.
2. Modify file paths to point to your input images and desired output locations.
3. Execute the script:
    ```bash
    python image_processing.py
    ```
4. View output shapes and images.

---

## 📝 Customization
- Modify:
  - **File paths** for input/output images.
  - **Gaussian kernel size** for blurring.
  - **Canny edge detection thresholds**.
  - **Hough Circle detection parameters**.

---

## 📁 Example Directory Structure

- project-folder/
├── image_processing.py # Python script
├── README.md # This file
└── images/
├── input.jpg # Your input image
├── deblurred.png # Deblurred output
├── contour_img.png # Edge detection result
└── shapes_detected.png # Shape detection result

## 📢 Notes
- Results include:
  - Number of detected shapes printed in the console.
  - Optional saving of processed images (uncomment `cv2.imwrite` lines).

---

## 📜 License
This project is licensed under the MIT License.

