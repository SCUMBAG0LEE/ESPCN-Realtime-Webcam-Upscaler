import os
import cv2
import sys
import time
import torch
#import openvino 
#import tensorrt
import threading
import numpy as np
import pyvirtualcam
import torch.nn as nn  
from pathlib import Path
import onnxruntime as ort
from datetime import datetime
from pyvirtualcam import PixelFormat
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QTabWidget, QComboBox, QCheckBox, QFileDialog,
    QLineEdit, QTextEdit, QSizePolicy, QFrame, QRadioButton, QButtonGroup,
    QMessageBox
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings

# --- Constants ---
MIKU_TEAL = "#33CCCC"
MIKU_PINK = "#FF6699"
DARK_BG = "#2D2D2D"
LIGHT_FG = "#CCCCCC"
DARK_FG = "#1E1E1E"
WHITE = "#FFFFFF"
BORDER_COLOR = "#555555"
MAX_FILE_RESOLUTIONS = {
    "Unlimited": None,
    "1080p (1920x1080)": (1920, 1080),
    "720p (1280x720)": (1280, 720),
    "540p (960x540)": (960, 540)
}
APP_NAME = "ESPCNUpscaler"
ORG_NAME = "WGProjects"

# --- ONNX Helper Functions ---

def get_available_providers():
    """Gets available ONNX Runtime execution providers, prioritizing GPU."""
    providers = ['CPUExecutionProvider']
    available = ort.get_available_providers()
    if 'TensorrtExecutionProvider' in available:
        providers.insert(0, 'TensorrtExecutionProvider')
    if 'CUDAExecutionProvider' in available:
        pos = 1 if 'TensorrtExecutionProvider' in providers else 0
        providers.insert(pos, 'CUDAExecutionProvider')
    if 'DmlExecutionProvider' in available:
        pos = providers.index('CPUExecutionProvider')
        providers.insert(pos, 'DmlExecutionProvider')
    return available

def initialize_onnx_session(model_path, provider):
    """Initializes ONNX Runtime session with optimized threading."""
    print(f"Initializing ONNX session: {model_path} with provider: {provider}")
    try:
        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)

        print(f"Setting ONNX intra_op_num_threads: {session_options.intra_op_num_threads}")
        print(f"Setting ONNX inter_op_num_threads: {session_options.inter_op_num_threads}")

        session = ort.InferenceSession(model_path, sess_options=session_options, providers=[provider])
        print(f"ONNX session initialized successfully with provider: {session.get_providers()}")
        return session
    except Exception as e:
        print(f"Error initializing ONNX session: {e}")
        return None

# --- Image Processing ---
def preprocess_image(frame, is_fp16=False):
    """Prepares a frame (NumPy array BGR HWC) for inference."""
    dtype = np.float16 if is_fp16 else np.float32
    img = frame.astype(dtype) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension -> NCHW
    return img

def postprocess_output(output_tensor, is_output_rgb=True):
    """Converts output tensor back to displayable image (NumPy array BGR HWC)."""
    if output_tensor is None or output_tensor.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    try:
        # Handle torch.Tensor or np.ndarray
        if isinstance(output_tensor, torch.Tensor):
            output_tensor = output_tensor.cpu().numpy()

        output_image = output_tensor.squeeze()
        if output_image.ndim == 2:
             output_image = np.expand_dims(output_image, axis=0)

        if output_image.shape[0] in [1, 3]:
            output_image = output_image.transpose(1, 2, 0)
        
        # Handle FP16 output
        if output_image.dtype == np.float16:
            output_image = output_image.astype(np.float32)

        output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)

        if is_output_rgb and output_image.shape[2] == 3:
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        elif output_image.shape[2] == 1:
             output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)

        return output_image
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8)

# --- PyTorch Model Definition ---
class ESPCN(nn.Module):
    def __init__(self, upscale_factor):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3 * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU() 

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

# --- Unified Inference Engine Wrapper ---
class InferenceEngine:
    """A unified wrapper for PyTorch and ONNX Runtime inference."""
    def __init__(self, engine_type, model_path, provider, is_fp16, model_scale_factor):
        self.engine_type = engine_type
        self.is_fp16 = is_fp16
        self.model = None
        self.session = None

        if self.engine_type == 'pytorch':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Initializing PyTorch engine on {self.device}...")
            self.model = ESPCN(upscale_factor=model_scale_factor).to(self.device)
            
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
            except Exception as e:
                print(f"Error loading .pth file. Is it a model state_dict or a full checkpoint? {e}")
                raise RuntimeError(f"Failed to load PyTorch model: {e}")

            original_state_dict = checkpoint.get('model_state_dict', checkpoint if isinstance(checkpoint, dict) else None)
            if original_state_dict is None:
                original_state_dict = checkpoint

            new_state_dict = {}
            needs_prefix_stripping = any(key.startswith('_orig_mod.') for key in original_state_dict.keys())
            
            if needs_prefix_stripping:
                print("Stripping '_orig_mod.' prefix from PyTorch model...")
                for key, value in original_state_dict.items():
                    if key.startswith('_orig_mod.'):
                        new_key = key[len('_orig_mod.'):]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
            else:
                 new_state_dict = original_state_dict
            
            self.model.load_state_dict(new_state_dict)
            if self.is_fp16:
                self.model.half()
            self.model.eval()
            print("PyTorch model loaded successfully.")

        elif self.engine_type == 'onnx':
            print(f"Initializing ONNX engine with provider: {provider}...")
            self.session = initialize_onnx_session(model_path, provider)
            if self.session:
                self.input_name = self.session.get_inputs()[0].name
                print(f"ONNX session loaded successfully. Provider: {self.session.get_providers()}")
            else:
                raise RuntimeError("Failed to initialize ONNX session.")

    def run(self, numpy_input):
        """Runs inference on a NumPy input and returns a NumPy output."""
        if self.engine_type == 'pytorch':
            input_tensor = torch.from_numpy(numpy_input).to(self.device)
            # PyTorch autocast for FP16
            with torch.no_grad(), torch.autocast(device_type=self.device.lower(), enabled=self.is_fp16):
                output_tensor = self.model(input_tensor)
            return output_tensor.cpu().numpy()
        
        elif self.engine_type == 'onnx' and self.session:
            ort_inputs = {self.input_name: numpy_input}
            ort_outs = self.session.run(None, ort_inputs)
            return ort_outs[0]
        
        else:
            raise RuntimeError("Inference engine is not properly initialized.")

# --- CameraReaderThread ---
class CameraReaderThread(QThread):
    """Dedicated thread to continuously read frames from the camera."""
    error_signal = pyqtSignal(str)

    def __init__(self, camera_index, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def run(self):
        self.running = True
        print(f"CameraReaderThread started for index {self.camera_index}")
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            error_msg = f"Error: CameraReaderThread could not open camera {self.camera_index}"
            print(error_msg)
            self.error_signal.emit(error_msg)
            self.running = False
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"ReaderThread: Attempted 1280x720, got {actual_width}x{actual_height}, BufferSize={cap.get(cv2.CAP_PROP_BUFFERSIZE)}")

        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
            else:
                time.sleep(0.01)

        cap.release()
        print("CameraReaderThread finished.")

    def get_latest_frame(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            else:
                return None

    def stop(self):
        print("Stopping CameraReaderThread...")
        self.running = False

# --- WebcamThread ---
class WebcamThread(QThread):
    update_frame_signal = pyqtSignal(np.ndarray, np.ndarray, dict)
    finished_signal = pyqtSignal()
    camera_error_signal = pyqtSignal(str)

    def __init__(self, camera_index, inference_engine, frameskip_value, resolution_mode,
                 model_input_size, model_scale_factor, is_output_rgb, is_fp16, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.inference_engine = inference_engine 
        self.running = False
        self.preview_enabled = True
        self.virtual_cam_enabled = False

        self.frameskip_value = max(1, frameskip_value)
        self.frame_counter = 0
        self.last_processed_frame = None
        self.last_stats = {}
        self.current_original_frame = None

        self.resolution_mode = resolution_mode
        self.model_input_size = model_input_size
        self.model_scale_factor = model_scale_factor
        self.is_output_rgb = is_output_rgb
        self.is_fp16 = is_fp16

        self.target_input_size = None
        self.output_size = None

        if self.model_input_size:
            self.target_input_size = self.model_input_size
            print(f"Webcam using FIXED model input size: {self.target_input_size}")
        else:
            mode_map = {"360p Input": (640, 360), "540p Input": (960, 540), "720p Input": (1280, 720)}
            self.target_input_size = mode_map.get(self.resolution_mode, (640, 360))
            print(f"Webcam using DYNAMIC model input size (from mode): {self.target_input_size}")

        if self.target_input_size and self.model_scale_factor:
             self.output_size = (self.target_input_size[0] * self.model_scale_factor,
                                 self.target_input_size[1] * self.model_scale_factor)
             print(f"Webcam calculated output size: {self.output_size}")
        else:
             print("Warning: Could not determine output size reliably.")
             self.output_size = (1280, 720)

        self.reader_thread = None

    def run(self):
        cv2.setNumThreads(1)
        self.running = True

        self.reader_thread = CameraReaderThread(self.camera_index)
        self.reader_thread.error_signal.connect(self.camera_error_signal.emit)
        self.reader_thread.start()

        virt_cam = None
        if self.virtual_cam_enabled and self.output_size:
            try:
                virt_cam = pyvirtualcam.Camera(
                    width=self.output_size[0], height=self.output_size[1], fps=30,
                    fmt=PixelFormat.BGR
                )
                print(f"Virtual camera started: {virt_cam.device} @ {virt_cam.width}x{virt_cam.height}")
            except Exception as e:
                print(f"Could not start virtual camera: {e}")
                self.virtual_cam_enabled = False

        print(f"Webcam processing thread started... Processing 1 frame every {self.frameskip_value}")

        while self.running:
            t_start_frame = time.perf_counter()

            if not self.reader_thread or not self.reader_thread.isRunning():
                 if self.running:
                     print("Camera reader thread stopped unexpectedly.")
                     self.camera_error_signal.emit("Camera disconnected or stopped working.")
                 break

            frame = self.reader_thread.get_latest_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            t_capture = time.perf_counter()

            self.current_original_frame = frame
            original_frame_for_display = frame

            self.frame_counter += 1
            stats = {}
            frame_to_display = None

            if self.frame_counter % self.frameskip_value == 0:
                t_start_pre = time.perf_counter()
                lr_frame = cv2.resize(frame, self.target_input_size, interpolation=cv2.INTER_AREA)
                input_tensor = preprocess_image(lr_frame, self.is_fp16)
                t_preproc = time.perf_counter()

                t_start_inf = time.perf_counter()
                output_tensor = self.inference_engine.run(input_tensor)
                t_inference = time.perf_counter()

                t_start_post = time.perf_counter()
                processed_frame = postprocess_output(output_tensor, self.is_output_rgb)
                t_postproc = time.perf_counter()

                self.last_processed_frame = processed_frame
                frame_to_display = processed_frame
                t_end_frame = time.perf_counter()

                stats = {
                    "get_frame_ms": (t_capture - t_start_frame) * 1000,
                    "preprocess_ms": (t_preproc - t_start_pre) * 1000,
                    "inference_ms": (t_inference - t_start_inf) * 1000,
                    "postprocess_ms": (t_postproc - t_start_post) * 1000,
                    "total_frame_ms": (t_end_frame - t_start_frame) * 1000,
                    "fps": 1.0 / (t_end_frame - t_start_frame) if (t_end_frame - t_start_frame) > 0 else 0,
                    "status": "PROCESSED"
                }
                self.last_stats = stats
            else:
                if self.last_processed_frame is not None:
                    frame_to_display = self.last_processed_frame
                    stats = self.last_stats.copy()
                    t_end_frame = time.perf_counter()
                    total_ms = (t_end_frame - t_start_frame) * 1000
                    stats["total_frame_ms"] = total_ms
                    stats["fps"] = 1000.0 / total_ms if total_ms > 0 else 0
                    stats["status"] = f"SKIPPED ({self.frame_counter % self.frameskip_value}/{self.frameskip_value - 1})"
                else:
                    continue

            if self.virtual_cam_enabled and frame_to_display is not None and virt_cam:
                try:
                    virt_cam.send(frame_to_display)
                    virt_cam.sleep_until_next_frame()
                except Exception as e:
                    print(f"Error sending frame to virtual camera: {e}")

            if frame_to_display is not None and self.running:
                input_frame_to_emit = original_frame_for_display if self.preview_enabled else np.zeros((10, 10, 3), dtype=np.uint8)
                output_frame_to_emit = frame_to_display if self.preview_enabled else np.zeros((10, 10, 3), dtype=np.uint8)
                self.update_frame_signal.emit(input_frame_to_emit, output_frame_to_emit, stats)

        if self.reader_thread:
            self.reader_thread.stop()
            self.reader_thread.wait()
            print("CameraReaderThread joined.")
        if virt_cam:
            print("Closing virtual camera...")
            virt_cam.close()

        print("Webcam processing thread finished.")
        self.finished_signal.emit()

    def stop(self):
        print("Stopping webcam processing thread...")
        self.running = False

    def get_current_frames(self):
        return self.current_original_frame, self.last_processed_frame

# --- FileProcessThread ---
class FileProcessThread(QThread):
    finished_signal = pyqtSignal(np.ndarray, np.ndarray, float)
    error_signal = pyqtSignal(str)

    def __init__(self, input_path, main_window, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.main_window = main_window
        self.inference_engine = self.main_window.inference_engine
        if not self.inference_engine: raise ValueError("Inference Engine not available for FileProcessThread")
        self.running = True

    def run(self):
        try:
            start_time = time.time()
            print(f"Processing file: {self.input_path}")
            if not self.running: return

            original_img = cv2.imread(self.input_path)
            if original_img is None: raise ValueError("Could not read input image file.")
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = original_img.shape[:2]

            img_to_process = original_img
            status_suffix = ""

            target_input_size = self.main_window.model_input_size

            if target_input_size:
                print(f"File using FIXED model input size: {target_input_size}")
                interpolation = cv2.INTER_AREA if (w_orig > target_input_size[0] or h_orig > target_input_size[1]) else cv2.INTER_CUBIC
                img_to_process = cv2.resize(original_img, target_input_size, interpolation=interpolation)
                status_suffix = f" (Resized to {target_input_size[0]}x{target_input_size[1]})"
            else:
                max_res_setting = self.main_window.max_file_res_combo.currentText()
                max_res_dims = MAX_FILE_RESOLUTIONS[max_res_setting]

                if max_res_dims and (w_orig > max_res_dims[0] or h_orig > max_res_dims[1]):
                    print(f"File input ({w_orig}x{h_orig}) exceeds limit ({max_res_dims[0]}x{max_res_dims[1]}). Downscaling...")
                    scale = min(max_res_dims[0] / w_orig, max_res_dims[1] / h_orig)
                    new_w = int(w_orig * scale)
                    new_h = int(h_orig * scale)
                    img_to_process = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    print(f"Downscaled to {new_w}x{new_h} for processing.")
                    status_suffix = f" (Downscaled to {new_w}x{new_h})"
                else:
                    print(f"File input ({w_orig}x{h_orig}) within limits or unlimited. Processing original size.")

            if not self.running: return
            print("Preprocessing...")
            input_tensor = preprocess_image(img_to_process, self.main_window.is_fp16)

            if not self.running: return
            print("Running inference...")
            # --- Unified inference call ---
            output_tensor = self.inference_engine.run(input_tensor)

            if not self.running: return
            print("Postprocessing...")
            upscaled_img_bgr = postprocess_output(output_tensor, self.main_window.is_output_rgb)
            upscaled_img_rgb = cv2.cvtColor(upscaled_img_bgr, cv2.COLOR_BGR2RGB)

            processing_time = time.time() - start_time
            print(f"File processing finished in {processing_time:.2f} seconds.")
            self.finished_signal.emit(original_img_rgb, upscaled_img_rgb, processing_time)

        except Exception as e:
            if self.running:
                print(f"Error during file processing: {e}")
                self.error_signal.emit(f"Error processing file: {e}")
        finally:
            self.running = False

    def stop(self):
        print("Requesting file processing cancellation...")
        self.running = False

# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESPCN Upscaler V9 (PyTorch/ONNX)") # V9: Updated title
        self.setGeometry(100, 100, 1000, 700)

        # --- State Variables ---
        self.inference_engine = None 
        self.webcam_thread = None
        self.file_thread = None
        self.current_model_path = ""
        self.current_provider = ""
        self.current_engine = ""
        self.is_fp16 = False
        self.is_output_rgb = True
        self.model_input_size = None
        self.model_scale_factor = 2

        self.webcam_input_pixmap = None
        self.webcam_output_pixmap = None
        self.original_pixmap = None
        self.upscaled_pixmap = None
        self.upscaled_image_data = None

        self.snapshot_dir = Path("./outputs/snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.handle_resize)

        self.settings = QSettings(ORG_NAME, APP_NAME)

        # --- UI Setup ---
        self.setStyleSheet(self.get_miku_stylesheet())
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.init_webcam_tab()
        self.init_file_tab()
        self.init_settings_tab()

        self.load_settings()
        self.reinitialize_session() 

        sys.stdout = self
        sys.stderr = self
        print(f"--- {APP_NAME} V9 Initialized ---")
        print(f"Available ONNX Providers: {self.available_providers}")

    def get_miku_stylesheet(self):
        return f"""
            QMainWindow {{ background-color: {DARK_BG}; }}
            QTabWidget::pane {{ border-top: 2px solid {BORDER_COLOR}; background-color: {DARK_BG}; }}
            QTabBar::tab {{ background: {DARK_FG}; color: {LIGHT_FG}; border: 1px solid {BORDER_COLOR}; border-bottom: none; padding: 8px 15px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; min-width: 100px; }}
            QTabBar::tab:selected {{ background: {DARK_BG}; color: {MIKU_TEAL}; border-color: {BORDER_COLOR}; border-bottom-color: {DARK_BG}; font-weight: bold; }}
            QTabBar::tab:!selected {{ margin-top: 2px; }}
            QTabBar::tab:hover {{ background: #444444; color: {MIKU_PINK}; }}
            QWidget {{ background-color: {DARK_BG}; color: {LIGHT_FG}; font-family: Noto Sans, Arial, sans-serif; font-size: 10pt; }}
            QLabel {{ color: {LIGHT_FG}; padding: 5px; }}
            QLabel#VideoLabel, QLabel#OriginalImageLabel, QLabel#UpscaledImageLabel {{ background-color: black; border: 1px solid {BORDER_COLOR}; padding: 0px; }}
            QLabel#StatsLabel {{ font-family: Consolas, Courier New, monospace; font-size: 9pt; border: 1px solid {BORDER_COLOR}; background-color: {DARK_FG}; padding: 5px; border-radius: 3px; }}
            QPushButton {{ background-color: {MIKU_TEAL}; color: {DARK_FG}; border: 1px solid {MIKU_TEAL}; padding: 8px 15px; border-radius: 4px; font-weight: bold; }}
            QPushButton:hover {{ background-color: #40E0D0; }}
            QPushButton:pressed {{ background-color: #20B2AA; }}
            QPushButton:disabled {{ background-color: #555555; color: #999999; border-color: #444444; }}
            QComboBox {{ background-color: {DARK_FG}; color: {LIGHT_FG}; border: 1px solid {BORDER_COLOR}; padding: 5px; border-radius: 3px; selection-background-color: {MIKU_TEAL}; selection-color: {DARK_FG}; }}
            QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right; width: 15px; border-left-width: 1px; border-left-color: {BORDER_COLOR}; border-left-style: solid; border-top-right-radius: 3px; border-bottom-right-radius: 3px; }}
            QComboBox QAbstractItemView {{ background-color: {DARK_FG}; border: 1px solid {BORDER_COLOR}; selection-background-color: {MIKU_TEAL}; selection-color: {DARK_FG}; color: {LIGHT_FG}; outline: none; }}
            QCheckBox {{ spacing: 5px; }}
            QCheckBox::indicator {{ width: 13px; height: 13px; border: 1px solid {MIKU_TEAL}; border-radius: 3px; background-color: {DARK_FG}; }}
            QCheckBox::indicator:checked {{ background-color: {MIKU_TEAL}; image: url(none); }}
            QCheckBox::indicator:unchecked:hover {{ border: 1px solid {MIKU_PINK}; }}
            QCheckBox::indicator:checked:hover {{ background-color: {MIKU_PINK}; border: 1px solid {MIKU_PINK}; }}
            QLineEdit, QTextEdit {{ background-color: {DARK_FG}; color: {LIGHT_FG}; border: 1px solid {BORDER_COLOR}; padding: 5px; border-radius: 3px; selection-background-color: {MIKU_TEAL}; selection-color: {DARK_FG}; }}
            QTextEdit {{ font-family: Consolas, Courier New, monospace; font-size: 9pt; }}
            QRadioButton {{ spacing: 5px; }}
            QRadioButton::indicator {{ width: 13px; height: 13px; border: 1px solid {MIKU_TEAL}; border-radius: 7px; background-color: {DARK_FG}; }}
            QRadioButton::indicator:checked {{ background-color: {MIKU_TEAL}; border: 2px solid {DARK_FG}; }}
            QRadioButton::indicator:unchecked:hover {{ border: 1px solid {MIKU_PINK}; }}
            QRadioButton::indicator:checked:hover {{ background-color: {MIKU_PINK}; border: 2px solid {DARK_FG}; }}
            QFrame#SeparatorLine {{ background-color: {BORDER_COLOR}; height: 1px; margin-top: 10px; margin-bottom: 10px; }}
        """

    # --- Tab Initializers ---
    def init_webcam_tab(self):
        self.webcam_tab = QWidget()
        layout = QHBoxLayout(self.webcam_tab)
        controls_layout = QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        cam_select_layout = QHBoxLayout()
        cam_select_layout.addWidget(QLabel("Camera:"))
        self.cam_combo = QComboBox()
        self.populate_cameras()
        cam_select_layout.addWidget(self.cam_combo)
        controls_layout.addLayout(cam_select_layout)

        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Mode (Dynamic Only):"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["360p Input", "540p Input", "720p Input"])
        self.resolution_combo.setEnabled(False)
        res_layout.addWidget(self.resolution_combo)
        controls_layout.addLayout(res_layout)

        self.start_button = QPushButton("Start Webcam")
        self.start_button.clicked.connect(self.start_webcam)
        self.stop_button = QPushButton("Stop Webcam")
        self.stop_button.clicked.connect(self.stop_webcam)
        self.stop_button.setEnabled(False)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        controls_layout.addLayout(button_layout)

        self.snapshot_button = QPushButton("Snapshot")
        self.snapshot_button.clicked.connect(self.take_webcam_snapshot)
        self.snapshot_button.setEnabled(False)
        controls_layout.addWidget(self.snapshot_button)

        self.preview_checkbox = QCheckBox("Show Preview")
        self.preview_checkbox.setChecked(True)
        self.preview_checkbox.stateChanged.connect(self.toggle_preview)
        controls_layout.addWidget(self.preview_checkbox)

        self.virtual_cam_checkbox = QCheckBox("Output to Virtual Camera")
        self.virtual_cam_checkbox.stateChanged.connect(self.toggle_virtual_cam)
        controls_layout.addWidget(self.virtual_cam_checkbox)

        frameskip_layout = QHBoxLayout()
        frameskip_layout.addWidget(QLabel("Process 1 frame every:"))
        self.frameskip_combo = QComboBox()
        self.frameskip_combo.addItems(["1 (No Skip)", "2 frames", "3 frames", "4 frames"])
        frameskip_layout.addWidget(self.frameskip_combo)
        controls_layout.addLayout(frameskip_layout)
        controls_layout.addStretch()

        video_stats_layout = QVBoxLayout()
        video_preview_layout = QHBoxLayout()

        input_video_layout = QVBoxLayout()
        input_video_label_text = QLabel("Input (Webcam)")
        input_video_label_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_video_label = QLabel("Webcam preview")
        self.input_video_label.setObjectName("VideoLabel")
        self.input_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        input_video_layout.addWidget(input_video_label_text)
        input_video_layout.addWidget(self.input_video_label, 1)
        video_preview_layout.addLayout(input_video_layout)

        output_video_layout = QVBoxLayout()
        self.output_video_label_text = QLabel(f"Output (Upscaled x{self.model_scale_factor})")
        self.output_video_label_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_video_label = QLabel("Upscaled output")
        self.output_video_label.setObjectName("VideoLabel")
        self.output_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        output_video_layout.addWidget(self.output_video_label_text)
        output_video_layout.addWidget(self.output_video_label, 1)
        video_preview_layout.addLayout(output_video_layout)

        video_stats_layout.addLayout(video_preview_layout, 1)

        self.stats_label = QLabel("Stats: Waiting for webcam...")
        self.stats_label.setObjectName("StatsLabel")
        self.stats_label.setFixedHeight(100)
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.stats_label.setWordWrap(True)
        video_stats_layout.addWidget(self.stats_label, 0)

        layout.addLayout(controls_layout, 0)
        layout.addLayout(video_stats_layout, 1)
        self.tabs.addTab(self.webcam_tab, "Webcam Upscaler")

    def init_file_tab(self):
        self.file_tab = QWidget()
        layout = QVBoxLayout(self.file_tab)
        top_controls_layout = QHBoxLayout()

        self.select_button = QPushButton("Select Image File")
        self.select_button.clicked.connect(self.select_image_file)
        top_controls_layout.addWidget(self.select_button)

        self.selected_file_label = QLabel("No file selected.")
        top_controls_layout.addWidget(self.selected_file_label, 1)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_file_processing)
        self.cancel_button.setEnabled(False)
        top_controls_layout.addWidget(self.cancel_button)

        self.save_button = QPushButton("Save Upscaled Image")
        self.save_button.clicked.connect(self.save_upscaled_image)
        self.save_button.setEnabled(False)
        top_controls_layout.addWidget(self.save_button)
        layout.addLayout(top_controls_layout)

        self.original_pixmap = None
        self.upscaled_pixmap = None
        self.upscaled_image_data = None

        comparison_layout = QHBoxLayout()
        original_layout = QVBoxLayout()
        original_layout.addWidget(QLabel("Original"))
        self.original_image_label = QLabel("Original image")
        self.original_image_label.setObjectName("OriginalImageLabel")
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        original_layout.addWidget(self.original_image_label, 1)
        comparison_layout.addLayout(original_layout)

        upscaled_layout = QVBoxLayout()
        self.upscaled_image_label_text = QLabel(f"Upscaled (x{self.model_scale_factor})")
        self.upscaled_image_label = QLabel("Upscaled image")
        self.upscaled_image_label.setObjectName("UpscaledImageLabel")
        self.upscaled_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.upscaled_image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        upscaled_layout.addWidget(self.upscaled_image_label_text)
        upscaled_layout.addWidget(self.upscaled_image_label, 1)
        comparison_layout.addLayout(upscaled_layout)
        layout.addLayout(comparison_layout, 1)

        self.file_status_label = QLabel("Status: Ready")
        layout.addWidget(self.file_status_label)
        self.tabs.addTab(self.file_tab, "File Upscaler")

    # --- init_settings_tab ---
    def init_settings_tab(self):
        self.settings_tab = QWidget()
        layout = QVBoxLayout(self.settings_tab)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- Engine Selection ---
        engine_layout = QHBoxLayout()
        engine_layout.addWidget(QLabel("Inference Engine:"))
        self.pytorch_radio = QRadioButton("PyTorch")
        self.onnx_radio = QRadioButton("ONNX Runtime")
        self.engine_group = QButtonGroup(self)
        self.engine_group.addButton(self.pytorch_radio, 0)
        self.engine_group.addButton(self.onnx_radio, 1)
        self.onnx_radio.setChecked(True) # Default to ONNX
        engine_layout.addWidget(self.pytorch_radio)
        engine_layout.addWidget(self.onnx_radio)
        engine_layout.addStretch()
        self.engine_group.idClicked.connect(self.reinitialize_session)
        self.engine_group.idClicked.connect(self.toggle_provider_combo)
        layout.addLayout(engine_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Path:"))
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select your .onnx or .pth model file")
        self.model_path_edit.editingFinished.connect(self.reinitialize_session)
        model_layout.addWidget(self.model_path_edit, 1)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_model_path)
        model_layout.addWidget(browse_button)
        layout.addLayout(model_layout)

        output_format_layout = QHBoxLayout()
        output_format_layout.addWidget(QLabel("Model Output Format:"))
        self.rgb_radio = QRadioButton("RGB (Default)")
        self.bgr_radio = QRadioButton("BGR")
        self.output_format_group = QButtonGroup(self)
        self.output_format_group.addButton(self.rgb_radio, 0)
        self.output_format_group.addButton(self.bgr_radio, 1)
        self.rgb_radio.setChecked(True)
        output_format_layout.addWidget(self.rgb_radio)
        output_format_layout.addWidget(self.bgr_radio)
        output_format_layout.addStretch()
        self.output_format_group.idClicked.connect(self.reinitialize_session)
        layout.addLayout(output_format_layout)

        precision_layout = QHBoxLayout()
        precision_layout.addWidget(QLabel("Model Precision:"))
        self.fp32_radio = QRadioButton("FP32")
        self.fp16_radio = QRadioButton("FP16")
        self.precision_group = QButtonGroup(self)
        self.precision_group.addButton(self.fp32_radio, 0)
        self.precision_group.addButton(self.fp16_radio, 1)
        self.fp32_radio.setChecked(True)
        precision_layout.addWidget(self.fp32_radio)
        precision_layout.addWidget(self.fp16_radio)
        precision_layout.addStretch()
        self.precision_group.idClicked.connect(self.reinitialize_session)
        layout.addLayout(precision_layout)

        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("Execution Provider (ONNX Only):"))
        self.provider_combo = QComboBox()
        self.available_providers = get_available_providers()
        self.provider_combo.addItems(self.available_providers)
        self.provider_combo.currentTextChanged.connect(self.reinitialize_session)
        provider_layout.addWidget(self.provider_combo, 1)
        layout.addLayout(provider_layout)

        max_res_layout = QHBoxLayout()
        max_res_layout.addWidget(QLabel("Max File Input Res (Dynamic):"))
        self.max_file_res_combo = QComboBox()
        self.max_file_res_combo.addItems(MAX_FILE_RESOLUTIONS.keys())
        self.max_file_res_combo.setCurrentText("1080p (1920x1080)")
        self.max_file_res_combo.setEnabled(False)
        max_res_layout.addWidget(self.max_file_res_combo, 1)
        layout.addLayout(max_res_layout)

        separator = QFrame()
        separator.setObjectName("SeparatorLine")
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        layout.addWidget(QLabel("Debug Log:"))
        self.debug_log = QTextEdit()
        self.debug_log.setReadOnly(True)
        layout.addWidget(self.debug_log, 1)
        self.tabs.addTab(self.settings_tab, "Settings & Debug")
    
    # --- Helper method ---
    def _parse_scale_from_path(self, path_str):
        """Tries to guess scale factor from filename (e.g., espcn_x2.pth)."""
        name = Path(path_str).stem.lower() # e.g., "best_espcn_x2_fp16"
        parts = name.split('_')
        for part in parts:
            if part.startswith('x') and len(part) > 1 and part[1].isdigit():
                try:
                    return int(part[1:])
                except ValueError:
                    continue
        print("Warning: Could not parse scale factor from filename. Defaulting to 2.")
        return 2 # Default fallback

    # --- Slot for radio buttons ---
    def toggle_provider_combo(self):
        """Enables/disables the ONNX Provider dropdown based on engine."""
        use_onnx = self.onnx_radio.isChecked()
        self.provider_combo.setEnabled(use_onnx)
        print(f"Provider combo enabled: {use_onnx}")

    # --- Settings Handlers ---
    # --- browse_model_path ---
    def browse_model_path(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Model File', '.', "Model Files (*.onnx *.pth)")
        if fname:
            self.model_path_edit.setText(fname)
            
            # --- Auto-select engine based on file type ---
            if fname.endswith('.pth'):
                self.pytorch_radio.setChecked(True)
                print("PyTorch model selected, auto-switching engine.")
            elif fname.endswith('.onnx'):
                self.onnx_radio.setChecked(True)
                print("ONNX model selected, auto-switching engine.")
            
            self.toggle_provider_combo() # Update provider box state
            self.reinitialize_session() # Re-init after selection

    # --- reinitialize_session ---
    def reinitialize_session(self):
        """Stops threads, reads settings, reloads the inference engine,
           parses metadata, updates UI, saves settings, and auto-restarts webcam."""
        print("Attempting to reinitialize inference engine...")
        was_running = self.webcam_thread and self.webcam_thread.isRunning()
        self.stop_webcam()
        self.cancel_file_processing()

        # --- Read current settings from UI ---
        model_path = self.model_path_edit.text()
        provider = self.provider_combo.currentText()
        is_fp16 = self.fp16_radio.isChecked()
        is_output_rgb = self.rgb_radio.isChecked()
        engine_type = 'pytorch' if self.pytorch_radio.isChecked() else 'onnx'

        # --- Prevent Redundant Reloads ---
        if (model_path == self.current_model_path and
            provider == self.current_provider and
            is_fp16 == self.is_fp16 and
            is_output_rgb == self.is_output_rgb and
            self.inference_engine is not None and
            self.current_engine == engine_type):
            print("Settings unchanged. Skipping reinitialization.")
            if was_running: self.start_webcam()
            return

        # --- Validations ---
        if not model_path: self.log_message("Model path cannot be empty."); return
        if not Path(model_path).exists():
            msg = f"Model file not found:\n{model_path}"
            self.log_message(f"Error: {msg}")
            QMessageBox.critical(self, "File Not Found", msg)
            return

        # Auto-select engine if file path mismatches radio button
        if model_path.endswith('.pth') and engine_type == 'onnx':
            print("Model path is .pth, but ONNX is selected. Switching to PyTorch.")
            engine_type = 'pytorch'
            self.pytorch_radio.setChecked(True)
        elif model_path.endswith('.onnx') and engine_type == 'pytorch':
            print("Model path is .onnx, but PyTorch is selected. Switching to ONNX.")
            engine_type = 'onnx'
            self.onnx_radio.setChecked(True)

        self.toggle_provider_combo() # Update UI state based on engine
        
        # Validate provider *only* if using ONNX
        if engine_type == 'onnx':
            if not provider:
                 if self.available_providers: provider = self.available_providers[0]; self.provider_combo.setCurrentText(provider)
                 else: self.log_message("Error: No execution provider available."); return
            if provider not in self.available_providers:
                msg = f"Selected provider '{provider}' is not available."
                self.log_message(f"Error: {msg}")
                QMessageBox.critical(self, "Provider Error", msg)
                return
        
        # --- Initialize Engine (PyTorch or ONNX) ---
        print(f"Selected Engine: {engine_type.upper()}")
        print(f"Selected Precision: {'FP16' if is_fp16 else 'FP32'}")
        print(f"Selected Output Format: {'RGB' if is_output_rgb else 'BGR'}")

        try:
            temp_model_scale_factor = 2 # Default
            temp_model_input_size = None # Default (Dynamic)

            if engine_type == 'onnx':
                print("Parsing ONNX metadata...")
                # Use a temp session to parse metadata without affecting the main one
                temp_session = initialize_onnx_session(model_path, provider)
                if not temp_session:
                    raise RuntimeError("Failed to create temporary ONNX session for metadata parsing.")
                
                input_meta = temp_session.get_inputs()[0]
                output_meta = temp_session.get_outputs()[0]
                self.log_message(f"  Input: {input_meta.name}, Shape: {input_meta.shape}, Type: {input_meta.type}")
                self.log_message(f"  Output: {output_meta.name}, Shape: {output_meta.shape}, Type: {output_meta.type}")

                input_shape = input_meta.shape; output_shape = output_meta.shape
                if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                    temp_model_input_size = (input_shape[3], input_shape[2]) # (w, h)
                
                try:
                    h_in, w_in = input_shape[2], input_shape[3]
                    h_out, w_out = output_shape[2], output_shape[3]
                    if isinstance(h_in, int) and isinstance(h_out, int) and h_in > 0:
                        temp_model_scale_factor = h_out // h_in
                    elif isinstance(w_in, int) and isinstance(w_out, int) and w_in > 0:
                        temp_model_scale_factor = w_out // w_in
                    else:
                        temp_model_scale_factor = self._parse_scale_from_path(model_path)
                except Exception:
                    temp_model_scale_factor = self._parse_scale_from_path(model_path)
                
                del temp_session 
            
            elif engine_type == 'pytorch':
                print("Parsing PyTorch metadata from filename...")
                temp_model_scale_factor = self._parse_scale_from_path(model_path)
                temp_model_input_size = None # PyTorch models are dynamic by default
            
            # --- Update UI with metadata ---
            self.model_scale_factor = temp_model_scale_factor
            self.model_input_size = temp_model_input_size
            self.log_message(f"  Detected Scale Factor: ~{self.model_scale_factor}x")
            
            if self.model_input_size:
                self.log_message(f"  Model has FIXED input size: {self.model_input_size}")
                self.resolution_combo.setEnabled(False)
                self.max_file_res_combo.setEnabled(False)
            else:
                self.log_message("  Model has DYNAMIC input size.")
                if not (self.webcam_thread and self.webcam_thread.isRunning()):
                    self.resolution_combo.setEnabled(True)
                self.max_file_res_combo.setEnabled(True)
            
            self.output_video_label_text.setText(f"Output (Upscaled x{self.model_scale_factor})")
            self.upscaled_image_label_text.setText(f"Upscaled (x{self.model_scale_factor})")
            
            # --- Final Engine Initialization ---
            self.inference_engine = InferenceEngine(
                engine_type, model_path, provider, is_fp16, self.model_scale_factor
            )
            
            # --- Store Successful State ---
            self.current_model_path = model_path
            self.current_provider = provider if engine_type == 'onnx' else 'N/A'
            self.is_fp16 = is_fp16
            self.is_output_rgb = is_output_rgb
            self.current_engine = engine_type # <-- NEW

            self.log_message(f"Successfully initialized {engine_type.upper()} engine.")
            self.start_button.setEnabled(True)
            self.select_button.setEnabled(True)
            self.save_settings()

            if was_running:
                print("Auto-restarting webcam with new engine...")
                self.start_webcam()

        except Exception as e:
            msg = f"Failed to initialize {engine_type.upper()} engine: {e}"
            self.log_message(msg)
            QMessageBox.critical(self, "Engine Initialization Error", msg)
            self.inference_engine = None; self.current_model_path = ""; self.current_engine = ""
            self.start_button.setEnabled(False); self.select_button.setEnabled(False)
            self.model_input_size = None; self.model_scale_factor = 2
            self.resolution_combo.setEnabled(False); self.max_file_res_combo.setEnabled(False)
            self.output_video_label_text.setText("Output (Upscaled x?)")
            self.upscaled_image_label_text.setText("Upscaled (x?)")

    # --- save_settings ---
    def save_settings(self):
        print("Saving settings...")
        try:
            self.settings.setValue("engine_type", "pytorch" if self.pytorch_radio.isChecked() else "onnx") # NEW
            self.settings.setValue("model_path", self.current_model_path)
            self.settings.setValue("provider", self.current_provider)
            self.settings.setValue("precision_fp16", self.is_fp16)
            self.settings.setValue("output_rgb", self.is_output_rgb)
            self.settings.setValue("max_file_res", self.max_file_res_combo.currentText())
            self.settings.setValue("preview_checked", self.preview_checkbox.isChecked())
            self.settings.setValue("virtualcam_checked", self.virtual_cam_checkbox.isChecked())
            self.settings.setValue("frameskip_index", self.frameskip_combo.currentIndex())
            if self.resolution_combo.isEnabled():
                self.settings.setValue("resolution_index", self.resolution_combo.currentIndex())
            self.settings.sync()
            print("Settings saved.")
        except Exception as e:
            print(f"Error saving settings: {e}")
            self.log_message(f"Warning: Could not save settings - {e}")

    # --- load_settings ---
    def load_settings(self):
        print("Loading settings...")
        try:
            # Load engine type first
            engine_type = self.settings.value("engine_type", "onnx")
            if engine_type == 'pytorch':
                self.pytorch_radio.setChecked(True)
            else:
                self.onnx_radio.setChecked(True)
            self.toggle_provider_combo() # Update provider UI based on loaded engine

            model_path = self.settings.value("model_path", "")
            self.model_path_edit.setText(model_path)

            default_provider = self.available_providers[0] if self.available_providers else ""
            provider = self.settings.value("provider", default_provider)
            if provider in self.available_providers: self.provider_combo.setCurrentText(provider)
            elif default_provider: self.provider_combo.setCurrentText(default_provider)

            is_fp16 = self.settings.value("precision_fp16", False, type=bool)
            if is_fp16: self.fp16_radio.setChecked(True)
            else: self.fp32_radio.setChecked(True)

            is_output_rgb = self.settings.value("output_rgb", True, type=bool)
            if is_output_rgb: self.rgb_radio.setChecked(True)
            else: self.bgr_radio.setChecked(True)

            max_res = self.settings.value("max_file_res", "1080p (1920x1080)")
            if max_res in MAX_FILE_RESOLUTIONS: self.max_file_res_combo.setCurrentText(max_res)

            self.preview_checkbox.setChecked(self.settings.value("preview_checked", True, type=bool))
            self.virtual_cam_checkbox.setChecked(self.settings.value("virtualcam_checked", False, type=bool))
            self.frameskip_combo.setCurrentIndex(self.settings.value("frameskip_index", 0, type=int))
            
            saved_res_index = self.settings.value("resolution_index", 0, type=int)
            if 0 <= saved_res_index < self.resolution_combo.count():
                 self.resolution_combo.setCurrentIndex(saved_res_index)

            print("Settings loaded.")
        except Exception as e:
            print(f"Error loading settings: {e}")
            self.log_message(f"Warning: Could not load settings - {e}")

    # --- Webcam Tab Slots ---
    def populate_cameras(self):
        available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(f"Camera {i}")
                cap.release()
        if available_cameras:
            self.cam_combo.addItems(available_cameras)
        else:
            self.cam_combo.addItem("No cameras found")
            self.cam_combo.setEnabled(False)
            self.start_button.setEnabled(False)

    # --- start_webcam ---
    def start_webcam(self):
        if not self.inference_engine:
            self.log_message("Error: Inference Engine not initialized.")
            return
        if self.webcam_thread and self.webcam_thread.isRunning():
            print("Webcam already running.")
            return
        cam_text = self.cam_combo.currentText()
        if "Camera" not in cam_text:
            self.log_message("No valid camera selected.")
            return

        cam_index = int(cam_text.split()[-1])
        frameskip_text = self.frameskip_combo.currentText()
        frameskip_value = int(frameskip_text.split()[0])
        resolution_mode = self.resolution_combo.currentText()
        print(f"Starting webcam {cam_index}: Frameskip={frameskip_value}, Mode={resolution_mode if not self.model_input_size else 'Fixed Model Input'}")

        self.webcam_thread = WebcamThread(
            cam_index, self.inference_engine, frameskip_value, resolution_mode,
            self.model_input_size, self.model_scale_factor, self.is_output_rgb, self.is_fp16
        )
        self.webcam_thread.update_frame_signal.connect(self.update_webcam_preview)
        self.webcam_thread.finished_signal.connect(self.webcam_finished_cleanup)
        self.webcam_thread.camera_error_signal.connect(self.handle_camera_error)
        self.webcam_thread.preview_enabled = self.preview_checkbox.isChecked()
        self.webcam_thread.virtual_cam_enabled = self.virtual_cam_checkbox.isChecked()
        self.webcam_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.cam_combo.setEnabled(False)
        self.resolution_combo.setEnabled(False)
        self.frameskip_combo.setEnabled(False)
        self.snapshot_button.setEnabled(True)

    def stop_webcam(self):
        if self.webcam_thread and self.webcam_thread.isRunning():
            print("Stopping webcam thread...")
            self.webcam_thread.stop()
        self.reset_webcam_ui_state()

    def webcam_finished_cleanup(self):
        print("Webcam processing thread reported finished.")
        self.webcam_thread = None
        self.webcam_input_pixmap = None
        self.webcam_output_pixmap = None
        self.reset_webcam_ui_state()

    def reset_webcam_ui_state(self):
        # --- reset_webcam_ui_state ---
        self.start_button.setEnabled(self.inference_engine is not None) 
        self.stop_button.setEnabled(False)
        self.snapshot_button.setEnabled(False)
        if self.cam_combo.count() > 0 and "No cameras found" not in self.cam_combo.itemText(0):
             self.cam_combo.setEnabled(True)
        self.resolution_combo.setEnabled(self.model_input_size is None and self.inference_engine is not None)
        self.frameskip_combo.setEnabled(True)

    def handle_camera_error(self, error_message):
        if self.webcam_thread:
             self.log_message(f"Camera Error: {error_message}")
             QMessageBox.warning(self, "Camera Error", error_message)
             self.stop_webcam()

    def take_webcam_snapshot(self):
        if self.webcam_thread and self.webcam_thread.isRunning():
            original_bgr, processed_bgr = self.webcam_thread.get_current_frames()
            if original_bgr is not None and processed_bgr is not None:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    self.snapshot_dir.mkdir(parents=True, exist_ok=True)
                    original_filename = self.snapshot_dir / f"snapshot_{timestamp}_original.png"
                    processed_filename = self.snapshot_dir / f"snapshot_{timestamp}_processed_x{self.model_scale_factor}.png"
                    cv2.imwrite(str(original_filename), original_bgr)
                    cv2.imwrite(str(processed_filename), processed_bgr)
                    self.log_message(f"Snapshot saved: {original_filename.name}, {processed_filename.name}")
                except Exception as e: self.log_message(f"Error saving snapshot: {e}")
            else: self.log_message("Could not get frames for snapshot.")
        else: self.log_message("Cannot take snapshot: Webcam is not running.")

    def update_webcam_preview(self, original_frame_bgr, processed_frame_bgr, stats):
        try:
            if self.webcam_thread and self.webcam_thread.preview_enabled:
                 h_orig, w_orig, _ = original_frame_bgr.shape
                 q_image_orig = QImage(np.ascontiguousarray(original_frame_bgr.data), w_orig, h_orig, 3 * w_orig, QImage.Format.Format_BGR888)
                 self.webcam_input_pixmap = QPixmap.fromImage(q_image_orig)
                 self.display_image(self.input_video_label, self.webcam_input_pixmap)

                 h_proc, w_proc, _ = processed_frame_bgr.shape
                 q_image_proc = QImage(np.ascontiguousarray(processed_frame_bgr.data), w_proc, h_proc, 3 * w_proc, QImage.Format.Format_BGR888)
                 self.webcam_output_pixmap = QPixmap.fromImage(q_image_proc)
                 self.display_image(self.output_video_label, self.webcam_output_pixmap)

            stats_text = (f"Status: {stats.get('status', 'N/A')} | "
                          f"Infer: {stats['inference_ms']:.1f} ms | "
                          f"Pre: {stats['preprocess_ms']:.1f} ms | "
                          f"Post: {stats['postprocess_ms']:.1f} ms\n"
                          f"Total Loop: {stats['total_frame_ms']:.1f} ms | "
                          f"Est. FPS: {stats['fps']:.1f}")
            self.stats_label.setText(stats_text)
        except Exception as e:
            print(f"Error updating webcam preview: {e}")

    def toggle_preview(self, state):
        is_checked = (state == Qt.CheckState.Checked.value)
        if self.webcam_thread and self.webcam_thread.isRunning():
            self.webcam_thread.preview_enabled = is_checked
            print(f"Preview toggled: {is_checked}")
        if not is_checked:
            self.input_video_label.clear(); self.output_video_label.clear()
            self.input_video_label.setText("Preview Disabled"); self.output_video_label.setText("Preview Disabled")
            self.webcam_input_pixmap = None; self.webcam_output_pixmap = None
        self.settings.setValue("preview_checked", is_checked)

    def toggle_virtual_cam(self, state):
        is_enabled = (state == Qt.CheckState.Checked.value)
        print(f"Virtual Cam output set to: {is_enabled} (will apply on next 'Start Webcam')")
        if self.webcam_thread and self.webcam_thread.isRunning():
            self.log_message("Virtual Cam setting will apply next time webcam is started.")
        self.settings.setValue("virtualcam_checked", is_enabled)

    # --- File Tab Slots ---
    # --- select_image_file ---
    def select_image_file(self):
        if not self.inference_engine:
            self.log_message("Error: Inference Engine not initialized.")
            return
        if self.file_thread and self.file_thread.isRunning():
            self.log_message("Warning: File processing in progress.")
            return

        fpath, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if fpath:
            self.selected_file_label.setText(Path(fpath).name)
            self.file_status_label.setText("Status: Processing...")
            self.original_image_label.setText("Loading..."); self.upscaled_image_label.setText("Processing...")
            self.save_button.setEnabled(False)
            self.select_button.setEnabled(False)
            self.cancel_button.setEnabled(True)

            self.file_thread = FileProcessThread(fpath, self)
            self.file_thread.finished_signal.connect(self.file_processing_finished)
            self.file_thread.error_signal.connect(self.file_processing_error)
            self.file_thread.finished.connect(self.file_processing_ui_cleanup)
            self.file_thread.start()

    def cancel_file_processing(self):
        if self.file_thread and self.file_thread.isRunning():
            self.file_thread.stop()
            self.file_status_label.setText("Status: Cancelling...")
            self.cancel_button.setEnabled(False)
        else:
            print("No file processing task to cancel.")

    def file_processing_finished(self, original_rgb, upscaled_rgb, processing_time):
        self.file_status_label.setText(f"Status: Finished in {processing_time:.2f} seconds.")
        try:
             self.upscaled_image_data = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)
             h_orig, w_orig, ch = original_rgb.shape
             q_image_orig = QImage(np.ascontiguousarray(original_rgb.data), w_orig, h_orig, ch * w_orig, QImage.Format.Format_RGB888)
             self.original_pixmap = QPixmap.fromImage(q_image_orig)
             self.display_image(self.original_image_label, self.original_pixmap)

             h_up, w_up, ch = upscaled_rgb.shape
             q_image_up = QImage(np.ascontiguousarray(upscaled_rgb.data), w_up, h_up, ch * w_up, QImage.Format.Format_RGB888)
             self.upscaled_pixmap = QPixmap.fromImage(q_image_up)
             self.display_image(self.upscaled_image_label, self.upscaled_pixmap)

             self.save_button.setEnabled(True)
        except Exception as e:
            self.file_processing_error(f"Error displaying results: {e}")

    def file_processing_error(self, error_message):
        self.log_message(f"File Error: {error_message}")
        QMessageBox.warning(self, "File Processing Error", error_message)
        self.file_status_label.setText(f"Status: Error")
        self.original_image_label.setText("Error"); self.upscaled_image_label.setText("Error")
        self.save_button.setEnabled(False)

    # --- file_processing_ui_cleanup ---
    def file_processing_ui_cleanup(self):
        self.file_thread = None
        self.select_button.setEnabled(self.inference_engine is not None) 
        self.cancel_button.setEnabled(False)
        current_status = self.file_status_label.text()
        if "Processing" in current_status or "Cancelling" in current_status:
            self.file_status_label.setText("Status: Cancelled or Ended")
            if not self.original_pixmap: self.original_image_label.setText("Cancelled")
            if not self.upscaled_pixmap: self.upscaled_image_label.setText("Cancelled")

    def display_image(self, label, pixmap):
        if pixmap and not label.size().isEmpty() and label.size().isValid():
             try:
                scaled_pixmap = pixmap.scaled(label.size(),
                                              Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
             except Exception as e:
                 print(f"Error scaling/displaying image: {e}")
                 label.setText("Display Error")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_timer.start(50)

    def handle_resize(self):
        self.display_image(self.original_image_label, self.original_pixmap)
        self.display_image(self.upscaled_image_label, self.upscaled_pixmap)
        self.display_image(self.input_video_label, self.webcam_input_pixmap)
        self.display_image(self.output_video_label, self.webcam_output_pixmap)

    def save_upscaled_image(self):
        if self.upscaled_image_data is None: self.log_message("No upscaled image data."); return
        original_name = Path(self.selected_file_label.text()).stem
        suggested_name = f"{original_name}_x{self.model_scale_factor}_upscaled.png"
        fpath, _ = QFileDialog.getSaveFileName(self, 'Save Upscaled Image', suggested_name, "PNG (*.png);;JPEG (*.jpg *.jpeg);;Bitmap (*.bmp)")
        if fpath:
            try:
                cv2.imwrite(fpath, self.upscaled_image_data)
                self.log_message(f"Upscaled image saved to {fpath}")
                self.file_status_label.setText(f"Status: Saved to {Path(fpath).name}")
            except Exception as e:
                self.log_message(f"Error saving image: {e}")
                self.file_status_label.setText("Status: Save Error")
                QMessageBox.critical(self, "Save Error", f"Could not save image:\n{e}")

    # --- Logging ---
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.debug_log.append(f"[{timestamp}] {message}")

    # --- QMainWindow Close Event ---
    def closeEvent(self, event):
        print("Closing application...")
        webcam_stopped = True
        file_stopped = True

        if self.webcam_thread and self.webcam_thread.isRunning():
            print("Waiting for webcam thread to stop...")
            self.webcam_thread.stop()
            if not self.webcam_thread.wait(1500):
                print("Webcam thread did not stop in time. Forcing termination.")
                self.webcam_thread.terminate()
                webcam_stopped = False

        if self.file_thread and self.file_thread.isRunning():
            print("Waiting for file thread to stop...")
            self.file_thread.stop()
            if not self.file_thread.wait(1500):
                print("File thread did not stop in time. Forcing termination.")
                self.file_thread.terminate()
                file_stopped = False

        if webcam_stopped and file_stopped:
            print("All threads stopped cleanly.")
        else:
            print("Warning: Some threads did not stop cleanly.")

        self.save_settings()
        event.accept()

    # --- Methods for redirecting print() ---
    def write(self, text):
        if text.strip():
             self.log_message(text.strip())

    def flush(self):
        pass

# --- Main Execution ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
