# ESPCN Realtime Webcam Upscaler

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

A desktop application for **real-time video super-resolution** using the ESPCN (Efficient Sub-Pixel Convolutional Neural Network) deep learning model. Upscales low-resolution webcam feeds (360p) to high-definition (720p) in real-time.

**Bachelor's Thesis Project:** *"Development Of An ESPCN Deep Learning Model For Real-Time Webcam Video Super-Resolution With Hardware Interference Optimization"*  
**Institution:** Universitas Tarumanagara, Faculty of Information Technology

---

## 🚀 Key Features

- **⚡ Real-Time Super-Resolution** — Upscales live video with low latency using ESPCN architecture
- **🎥 Virtual Camera Output** — Stream upscaled video to Zoom, Google Meet, Teams, or OBS
- **🖼️ Batch Image Processing** — Upscale static images with quality metrics comparison
- **⚙️ Multiple Inference Engines**
  - **ONNX Runtime** — TensorRT, CUDA, DirectML, OpenVINO providers
  - **PyTorch** — Native CUDA support with FP16/FP32 precision
- **🖥️ Modern GUI** — PyQt6 interface with persistent settings and easy model selection
- **📊 Quality Metrics** — PSNR, SSIM, MSE comparison tools included

---

## 📁 Project Structure

```
ESPCN-Realtime-Webcam-Upscaler/
├── Client.py                    # Main GUI application
├── requirements.txt             # Python dependencies
├── Models/                      # Pre-trained ESPCN models
│   ├── quality_espcn_x2.onnx    # Quality-focused model (ONNX)
│   ├── quality_espcn_x2.pth     # Quality-focused model (PyTorch)
│   ├── performance_espcn_x2.onnx # Performance-optimized model
│   └── *_fp16.*                 # FP16 quantized variants
├── Tools/                       # Utility scripts
│   ├── batch-process.py         # Batch image upscaling
│   ├── compare.py               # Image quality comparison (PSNR/SSIM)
│   ├── batch-compare.py         # Batch quality comparison
│   └── ...
└── Training Script/             # Model training notebooks
    ├── ESPCN_Training_Script_Gen_6_Windows.ipynb
    └── ESPCN_Training_Script_Gen_6_Colab.ipynb
```

---

## 🛠️ Requirements

- **OS:** Windows 10/11 or Linux
- **Python:** 3.10+
- **GPU:** NVIDIA GPU recommended (CUDA support)
  - *CPU-only mode available but significantly slower*

---

## 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ESPCN-Realtime-Webcam-Upscaler.git
   cd ESPCN-Realtime-Webcam-Upscaler
   ```

2. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv .venv
   .\.venv\Scripts\activate

   # Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** For NVIDIA GPU support, install `onnxruntime-gpu` instead of `onnxruntime-directml`:
   > ```bash
   > pip install onnxruntime-gpu
   > ```

---

## 💻 Usage

### Main Application
```bash
python Client.py
```

**Configuration:**
1. **Inference Engine** — Select `ONNX Runtime` (recommended) or `PyTorch`
2. **Provider** — Choose `TensorrtExecutionProvider` > `CUDAExecutionProvider` > `DmlExecutionProvider` > `CPUExecutionProvider`
3. **Model** — Load from `Models/` directory (`.onnx` or `.pth`)
4. **Precision** — FP16 for speed, FP32 for quality

**Webcam Mode:**
- Select camera source
- Enable "Output to Virtual Camera" for streaming apps
- Click **Start**

### Batch Processing Tools

```bash
# Batch upscale images
python Tools/batch-process.py

# Compare image quality (PSNR, SSIM, MSE)
python Tools/compare.py
```

---

## 🧠 Model Training

Training scripts for DIV2K + Flickr2K datasets are provided in `Training Script/`:

| Platform | Notebook |
|----------|----------|
| Windows (Local) | `ESPCN_Training_Script_Gen_6_Windows.ipynb` |
| Google Colab | `ESPCN_Training_Script_Gen_6_Colab.ipynb` |

**Training Features:**
- Combined MSE + SSIM loss function
- Mixed-precision training (FP16)
- Checkpoint resumption
- ONNX export with FP16 quantization
- Performance benchmarking

**Default Configuration:**
```python
UPSCALE_FACTOR = 2    # 2x super-resolution
BATCH_SIZE = 64       # Adjust for GPU memory
EPOCHS = 150
LEARNING_RATE = 1e-4
LOSS_ALPHA = 0.84     # MSE weight (SSIM = 0.16)
```

---

## 📊 Pre-trained Models

| Model | Type | Precision | Use Case |
|-------|------|-----------|----------|
| `quality_espcn_x2` | ONNX/PTH | FP32 | Best visual quality |
| `quality_espcn_x2_fp16` | ONNX/PTH | FP16 | Quality with faster inference |
| `performance_espcn_x2` | ONNX/PTH | FP32 | Balanced speed/quality |
| `performance_espcn_x2_fp16` | ONNX/PTH | FP16 | Maximum speed |

---

## 🔧 Tools

| Script | Description |
|--------|-------------|
| `batch-process.py` | Batch upscale images with ESPCN vs bicubic comparison |
| `compare.py` | Compare image quality metrics (MSE, PSNR, SSIM) |
| `batch-compare.py` | Batch quality comparison across multiple images |
| `batch-check.py` | Validate batch processing results |
| `inset.py` | Create comparison inset images |

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**William Gunawan**  
NPM: 535220092  
Universitas Tarumanagara, Faculty of Information Technology

---

## 📚 References

- Shi et al. (2016) — [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)
- DIV2K Dataset — [https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- Flickr2K Dataset — [https://cv.snu.ac.kr/research/EDSR/](https://cv.snu.ac.kr/research/EDSR/)
