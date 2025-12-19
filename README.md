# ESPCN Realtime Webcam Upscaler

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

**ESPCN Realtime Webcam Upscaler** is a desktop application designed to upscale low-resolution webcam video streams (e.g., 360p) to high-definition (e.g., 720p) in real-time using Deep Learning.

Developed as part of a Bachelor's Thesis: *"Development Of An ESPCN Deep Learning Model For Real-Time Webcam Video Super-Resolution With Hardware Interference Optimalization"* at Universitas Tarumanagara.

## 🚀 Key Features

* **⚡ Real-Time Super-Resolution**: Upscales live video feeds with low latency using the efficient ESPCN architecture.
* **🎥 Virtual Camera Output**: Pipes the upscaled video directly to Zoom, Google Meet, Teams, or OBS.
* **🖼️ Image Upscaler**: Drag-and-drop support for upscaling static images (`.png`, `.jpg`).
* **⚙️ Hardware Acceleration**:
    * **ONNX Runtime**: Supports TensorRT, CUDA, DirectML, and OpenVINO execution providers.
    * **PyTorch**: Native CUDA support.
* **🖥️ Modern GUI**: Built with PyQt6 for easy model selection and toggle controls.

## 🛠️ Requirements

* **OS**: Windows 10/11 or Linux
* **Python**: 3.10 or newer
* **GPU**: A dedicated NVIDIA GPU is recommended for optimal real-time performance.
    * *Compatible with CPU, though inference speed will be significantly lower.*

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/ESPCN-Realtime-Webcam-Upscaler.git](https://github.com/yourusername/ESPCN-Realtime-Webcam-Upscaler.git)
    cd ESPCN-Realtime-Webcam-Upscaler
    ```

2.  **Set Up Virtual Environment**
    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage Guide

1.  **Launch the App**
    ```bash
    python Client.py
    ```

2.  **Configuration (Settings Tab)**
    * **Inference Engine**: Choose `ONNX Runtime` for best speed.
    * **Provider**: Select `TensorrtExecutionProvider` (if available) or `CUDAExecutionProvider`.
    * **Model**: Load your `.onnx` or `.pth` model file (found in the `models/` directory).

3.  **Start Upscaling**
    * Navigate to the **Webcam Upscaler** tab.
    * Select your webcam source.
    * Check **"Output to Virtual Camera"** if you want to use the result in other applications.
    * Click **Start**.

## 🧠 Model Training

This project uses the **Efficient Sub-Pixel Convolutional Neural Network (ESPCN)**.
* **Dataset**: Trained on DIV2K and Flickr2K.
* **Training**: The full training pipeline is available in `ESPCN_Training_Script_Gen_6_Windows.ipynb`.

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## 👨‍💻 Author
**William Gunawan**
* **NPM**: 535220092
* **Institution**: Universitas Tarumanagara, Faculty of Information Technology
