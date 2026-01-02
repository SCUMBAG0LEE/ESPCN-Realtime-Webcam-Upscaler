"""
Batch Image Processing Script

Processes images through ESPCN upscaling model with comparison to bicubic interpolation.
Supports both PyTorch and ONNX Runtime inference engines.

Dependencies:
    - opencv-python
    - torch
    - numpy
    - onnxruntime
    - Pillow (optional)

Configuration:
    Update paths and model settings in the main() function before running.
"""

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import onnxruntime as ort

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class ESPCN(nn.Module):
    """Efficient Sub-Pixel Convolutional Neural Network for 2x upscaling."""
    
    def __init__(self, upscale_factor: int):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3 * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ESPCN network."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x


def preprocess_image(frame: np.ndarray, is_fp16: bool = False) -> np.ndarray:
    """Preprocess image for inference.
    
    Converts BGR HWC to NCHW float32/16, normalized to [0,1].
    
    Args:
        frame: Input image as NumPy array (BGR HWC).
        is_fp16: Whether to use FP16 precision.
    
    Returns:
        np.ndarray: Preprocessed tensor in NCHW format.
    """
    dtype = np.float16 if is_fp16 else np.float32
    img = frame.astype(dtype) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img


def postprocess_output(output_tensor: np.ndarray, is_output_rgb: bool = True) -> np.ndarray:
    """Postprocess model output to displayable image.
    
    Converts NCHW format back to BGR HWC uint8.
    
    Args:
        output_tensor: Model output tensor (NCHW).
        is_output_rgb: Whether output is RGB (needs BGR conversion).
    
    Returns:
        np.ndarray: Displayable image in BGR HWC format (uint8).
    """
    if output_tensor is None or output_tensor.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    if isinstance(output_tensor, torch.Tensor):
        output_tensor = output_tensor.cpu().numpy()

    output_image = output_tensor.squeeze()
    if output_image.ndim == 2:
        output_image = np.expand_dims(output_image, axis=0)
    if output_image.shape[0] in [1, 3]:
        output_image = output_image.transpose(1, 2, 0)  # CHW -> HWC

    if output_image.dtype == np.float16:
        output_image = output_image.astype(np.float32)

    output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)

    if is_output_rgb and output_image.shape[2] == 3:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    elif output_image.shape[2] == 1:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)

    return output_image


class InferenceEngine:
    """
    Unified wrapper for PyTorch and ONNX inference.
    For simplicity in this script, you will usually pick one engine.
    """

    def __init__(self,
                 engine_type: str,
                 model_path: str,
                 provider: str = "CPUExecutionProvider",
                 is_fp16: bool = False,
                 model_scale_factor: int = 2):
        self.engine_type = engine_type
        self.is_fp16 = is_fp16
        self.scale = model_scale_factor
        self.device = None
        self.model = None
        self.session = None

        if engine_type == "pytorch":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = ESPCN(upscale_factor=model_scale_factor).to(self.device)

            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Strip _orig_mod. prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_key = k[len("_orig_mod."):]
                else:
                    new_key = k
                new_state_dict[new_key] = v

            self.model.load_state_dict(new_state_dict)
            if is_fp16:
                self.model.half()
            self.model.eval()

        elif engine_type == "onnx":
            sess_opts = ort.SessionOptions()
            sess_opts.inter_op_num_threads = 1
            sess_opts.intra_op_num_threads = max(1, (os.cpu_count() or 1) // 2)
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_opts,
                providers=[provider]
            )
            self.input_name = self.session.get_inputs()[0].name
        else:
            raise ValueError("engine_type must be 'pytorch' or 'onnx'")

    def run(self, numpy_input: np.ndarray) -> np.ndarray:
        if self.engine_type == "pytorch":
            input_tensor = torch.from_numpy(numpy_input).to(self.device)
            with torch.no_grad(), torch.autocast(device_type=self.device, enabled=self.is_fp16):
                out = self.model(input_tensor)
            return out.cpu().numpy()
        else:
            ort_inputs = {self.input_name: numpy_input}
            ort_outs = self.session.run(None, ort_inputs)
            return ort_outs[0]


# =========================
# Dataset pipeline
# =========================

def make_even_dimensions(img: np.ndarray) -> np.ndarray:
    """Pad by at most 1 pixel on each side so H and W are divisible by 2."""
    h, w = img.shape[:2]
    new_w = w if w % 2 == 0 else w + 1
    new_h = h if h % 2 == 0 else h + 1

    if new_w == w and new_h == h:
        return img

    pad_right = new_w - w
    pad_bottom = new_h - h
    # Pad with replicated border so it is visually consistent
    img_padded = cv2.copyMakeBorder(
        img,
        top=0, bottom=pad_bottom,
        left=0, right=pad_right,
        borderType=cv2.BORDER_REPLICATE
    )
    return img_padded


def process_single_image(img_path: Path,
                         category: str,
                         idx: int,
                         out_root: Path,
                         engine: InferenceEngine,
                         scale: int = 2):
    """
    For one image:
    - ensure even dims
    - save GT
    - create LR (downscale)
    - bicubic upscale 2x
    - espcn upscale 2x
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: failed to read {img_path}")
        return

    # Ensure divisible-by-2 resolution
    img = make_even_dimensions(img)
    h, w = img.shape[:2]

    # Output folder: ./output/<Category>/<NN>
    idx_str = f"{idx:02d}"
    out_dir = out_root / category / idx_str
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{category}{idx_str}"

    # Ground truth (processed)
    gt_path = out_dir / f"{base_name}.png"
    cv2.imwrite(str(gt_path), img)

    # Downscale to half with bicubic
    lr_w, lr_h = w // scale, h // scale
    lr = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

    # Bicubic upscale back
    bicubic_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    bicubic_path = out_dir / f"{base_name}_bicubic{scale}x.png"
    cv2.imwrite(str(bicubic_path), bicubic_up)

    # ESPCN upscale back
    # ESPCN is trained for LR->HR, so feed LR
    lr_input = preprocess_image(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB), is_fp16=engine.is_fp16)
    espcn_out = engine.run(lr_input)
    espcn_img = postprocess_output(espcn_out, is_output_rgb=True)  # returns BGR
    # If model output size differs slightly, force resize to (w, h)
    if espcn_img.shape[1] != w or espcn_img.shape[0] != h:
        espcn_img = cv2.resize(espcn_img, (w, h), interpolation=cv2.INTER_CUBIC)

    espcn_path = out_dir / f"{base_name}_espcn{scale}x.png"
    cv2.imwrite(str(espcn_path), espcn_img)

    print(f"Done: {img_path} -> {out_dir}")


def collect_images(root: Path):
    """
    Returns list of (category, image_path) where category is direct child folder name.
    """
    items = []
    for category_dir in root.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for img_path in sorted(category_dir.iterdir()):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                items.append((category, img_path))
    return items


def main():
    # === CONFIG ===
    input_root = Path(r"C:\256GB\ESPCN\addition")  # your source
    output_root = Path("./results")  # your destination
    output_root.mkdir(exist_ok=True)

    model_path = r"checkpoints\quality_espcn_x2.onnx"  # or .onnx
    engine_type = "onnx"  # "pytorch" or "onnx"
    onnx_provider = "CUDAExecutionProvider"  # or "CPUExecutionProvider"
    is_fp16 = False
    scale = 2

    # === Init engine ===
    if engine_type == "pytorch":
        engine = InferenceEngine("pytorch", model_path, is_fp16=is_fp16, model_scale_factor=scale)
    else:
        engine = InferenceEngine("onnx", model_path, provider=onnx_provider,
                                 is_fp16=is_fp16, model_scale_factor=scale)

    # === Walk folders and process ===
    items = collect_images(input_root)

    # Group by category and give local indices 01..NN per category
    from collections import defaultdict
    per_cat = defaultdict(list)
    for cat, path in items:
        per_cat[cat].append(path)

    for cat, paths in per_cat.items():
        print(f"Category {cat}: {len(paths)} images")
        for i, img_path in enumerate(paths, start=1):
            process_single_image(
                img_path=img_path,
                category=cat,
                idx=i,
                out_root=output_root,
                engine=engine,
                scale=scale
            )


if __name__ == "__main__":
    main()
