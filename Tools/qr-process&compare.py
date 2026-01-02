"""
QR Code Robustness Testing Tool

Tests ESPCN upscaling robustness with QR codes.
Processes QR codes through downscaling/upscaling and verifies decodability.

Dependencies:
    - opencv-python
    - numpy
    - torch
    - pillow
    - scikit-image
    - matplotlib
    - pyzxing

Note:
    QR code decoding requires pyzxing with Java ZXing library installed.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

try:
    from pyzxing import BarCodeReader
except ImportError:
    print("Warning: pyzxing not installed. QR decoding will not work.")
    BarCodeReader = None

# ============================================================================
# ESPCN MODEL DEFINITION
# ============================================================================

class ESPCN(nn.Module):
    """Efficient Sub-Pixel Convolutional Neural Network for 2x upscaling."""
    
    def __init__(self, upscalefactor: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3 * (upscalefactor ** 2), kernel_size=3, padding=1)
        self.pixelshuffle = nn.PixelShuffle(upscalefactor)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ESPCN network."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixelshuffle(self.conv3(x))
        return x


def load_espcn_pytorch(model_path: str, device: torch.device, upscale: int = 2) -> nn.Module:
    model = ESPCN(upscalefactor=upscale).to(device)
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("orig_mod."):
            new_state[k[len("orig_mod."):]] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state)
    model.eval()
    return model


def espcn_upscale_bgr(lr_bgr: np.ndarray, model: nn.Module, device: torch.device, fp16: bool = False) -> np.ndarray:
    """
    Upscale a BGR image using ESPCN model (2x scale).
    
    Args:
        lr_bgr: Low-resolution input image in BGR format.
        model: ESPCN PyTorch model.
        device: Device to run model on (CPU/CUDA).
        fp16: Whether to use FP16 precision (only on CUDA).
    
    Returns:
        Upscaled 2x image in BGR format with shape (H*2, W*2, 3).
    """
    img = lr_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    tensor = torch.from_numpy(img).to(device)
    if fp16:
        tensor = tensor.half()

    use_autocast = fp16 and device.type == "cuda"

    with torch.no_grad():
        if use_autocast:
            with torch.autocast(device_type="cuda", enabled=True):
                out = model(tensor)
        else:
            out = model(tensor)

    out = out.squeeze(0).detach().cpu()
    if out.dtype == torch.float16:
        out = out.float()

    out = out.numpy()
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)
    out = np.transpose(out, (1, 2, 0))  # HWC, RGB

    if out.ndim != 3 or out.shape[2] != 3:
        h, w = out.shape[:2]
        out = np.zeros((h, w, 3), dtype=np.uint8)

    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


# ===================== ZXING QR HELPERS =====================

reader = BarCodeReader()

def qr_decode_info(img_path: str) -> Tuple[bool, str]:
    """
    Decode QR code from image using ZXing via pyzxing.
    
    Args:
        img_path: Path to image file.
    
    Returns:
        Tuple of (decoded_successfully, decoded_text).
        decoded_text is empty string if decoding failed or no text found.
    """
    try:
        results = reader.decode(img_path)
    except Exception:
        return False, ""

    if not results:
        return False, ""

    first = results[0]
    raw = first.get("parsed") or first.get("raw") or ""

    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = str(raw)
    else:
        text = str(raw)

    text = text.strip()
    return bool(text), text


def is_qr_decodable(img_path: str) -> bool:
    """
    Quick check if QR code in image is decodable.
    
    Args:
        img_path: Path to image file.
    
    Returns:
        True if QR code successfully decoded, False otherwise.
    """
    ok, _ = qr_decode_info(img_path)
    return ok


# ===================== COMPARISON (MSE/SSIM/QR) =====================

def compare_and_save(img_gt_file: str, img_bicubic_file: str, img_espcn_file: str, save_path: str) -> None:
    """
    Load and compare three images, calculate quality metrics and QR decodability.
    
    Creates a 3-panel matplotlib figure showing Ground Truth, Bicubic, and ESPCN
    with MSE, PSNR, SSIM metrics and QR decode status for each.
    
    Args:
        img_gt_file: Path to ground truth image.
        img_bicubic_file: Path to bicubic upscaled image.
        img_espcn_file: Path to ESPCN upscaled image.
        save_path: Where to save the comparison figure.
    """

    try:
        img_gt = io.imread(img_gt_file)
        img_bicubic = io.imread(img_bicubic_file)
        img_espcn = io.imread(img_espcn_file)
    except FileNotFoundError as e:
        print(f"Error: Cannot find file '{e.filename}'.")
        return
    except Exception as e:
        print(f"An error occurred while loading images: {e}")
        return

    # drop alpha
    if img_gt.ndim == 3 and img_gt.shape[-1] == 4:
        img_gt = img_gt[..., :3]
    if img_bicubic.ndim == 3 and img_bicubic.shape[-1] == 4:
        img_bicubic = img_bicubic[..., :3]
    if img_espcn.ndim == 3 and img_espcn.shape[-1] == 4:
        img_espcn = img_espcn[..., :3]

    if not (img_gt.shape == img_bicubic.shape == img_espcn.shape):
        print("Error: Images do not have the same dimensions. Cannot compare.")
        print(f"  GT:      {img_gt.shape}")
        print(f"  Bicubic: {img_bicubic.shape}")
        print(f"  ESPCN:   {img_espcn.shape}")
        return

    is_multichannel = img_gt.ndim == 3
    channel_axis = -1 if is_multichannel else None
    data_range = img_gt.max() - img_gt.min()

    mse_bicubic = mse(img_gt, img_bicubic)
    psnr_bicubic = psnr(img_gt, img_bicubic, data_range=data_range)
    ssim_bicubic = ssim(img_gt, img_bicubic, data_range=data_range, channel_axis=channel_axis)

    mse_espcn = mse(img_gt, img_espcn)
    psnr_espcn = psnr(img_gt, img_espcn, data_range=data_range)
    ssim_espcn = ssim(img_gt, img_espcn, data_range=data_range, channel_axis=channel_axis)

    gt_ok,  gt_text  = qr_decode_info(img_gt_file)
    bic_ok, bic_text = qr_decode_info(img_bicubic_file)
    esp_ok, esp_text = qr_decode_info(img_espcn_file)

    print("--- Image Quality & QR Comparison ---\n")
    print("Ground Truth:")
    print(f"  QR: {'SUCCESS' if gt_ok else 'FAIL'}  Text: {gt_text}")
    print()
    print("Bicubic:")
    print(f"  MSE:  {mse_bicubic:.2f}")
    print(f"  PSNR: {psnr_bicubic:.2f} dB")
    print(f"  SSIM: {ssim_bicubic:.4f}")
    print(f"  QR:   {'SUCCESS' if bic_ok else 'FAIL'}  Text: {bic_text}")
    print()
    print("ESPCN:")
    print(f"  MSE:  {mse_espcn:.2f}")
    print(f"  PSNR: {psnr_espcn:.2f} dB")
    print(f"  SSIM: {ssim_espcn:.4f}")
    print(f"  QR:   {'SUCCESS' if esp_ok else 'FAIL'}  Text: {esp_text}")
    print("\n----------------------------------")

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    def qr_status_block(ok, txt):
        status = "SUCCESS" if ok else "FAIL"
        if not ok or not txt:
            return f"QR: {status}"
        short = (txt[:40] + "…") if len(txt) > 40 else txt
        return f"QR: {status}\nText: {short}"

    # Ground Truth
    axes[0].imshow(img_gt)
    axes[0].set_title(
        "Ground Truth\n" +
        qr_status_block(gt_ok, gt_text),
        fontsize=10
    )
    axes[0].axis("off")

    # Bicubic
    axes[1].imshow(img_bicubic)
    axes[1].set_title(
        "Bicubic\n"
        f"MSE:  {mse_bicubic:.1f}\n"
        f"PSNR: {psnr_bicubic:.1f} dB\n"
        f"SSIM: {ssim_bicubic:.4f}\n" +
        qr_status_block(bic_ok, bic_text),
        fontsize=9
    )
    axes[1].axis("off")

    # ESPCN
    axes[2].imshow(img_espcn)
    axes[2].set_title(
        "ESPCN\n"
        f"MSE:  {mse_espcn:.1f}\n"
        f"PSNR: {psnr_espcn:.1f} dB\n"
        f"SSIM: {ssim_espcn:.4f}\n" +
        qr_status_block(esp_ok, esp_text),
        fontsize=9
    )
    axes[2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = Path(save_path)
    try:
        fig.savefig(save_path, dpi=150)
        print(f"Comparison image saved to: {save_path}")
    except Exception as e:
        print(f"Error saving comparison image: {e}")
    plt.close(fig)


# ===================== SWEET-SPOT SEARCH + EXPORT =====================

def find_qr_sweet_spot_for_image(
    img_path: Path,
    model: nn.Module,
    device: torch.device,
    min_size: int = 8,
    step: int = 2,
    fp16: bool = False,
) -> None:
    """
    Find optimal low-resolution size for QR robustness testing.
    
    Searches for the smallest LR size where ESPCN can decode the QR code
    but bicubic cannot. Exports GT, bicubic, and ESPCN images at sweet spot.
    
    Args:
        img_path: Path to input image with QR code.
        model: ESPCN PyTorch model for upscaling.
        device: Device to run model on.
        min_size: Minimum LR size to test.
        step: Size increment step.
        fp16: Use FP16 precision for inference.
    """
    if orig is None:
        print(f"[{img_path}] Could not read image, skipping.")
        return

    h, w = orig.shape[:2]
    side = min(h, w)
    if h != w:
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        orig = orig[y0:y0+side, x0:x0+side]
        h = w = side

    print(f"[{img_path}] Original square size: {w}x{h}")

    max_lr = max(min(w // 2, h // 2), min_size)
    print(f"[{img_path}] Scanning LR sizes from {min_size} up to {max_lr} (step {step})")

    found_strict = False
    found_any = False
    chosen_s = None
    chosen_results = None

    tmp_dir = img_path.parent / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    for s in range(min_size, max_lr + 1, step):
        lr = cv2.resize(orig, (s, s), interpolation=cv2.INTER_AREA)

        lr_path = tmp_dir / "lr.png"
        cv2.imwrite(str(lr_path), lr)

        if is_qr_decodable(str(lr_path)):
            print(f"  {s}x{s}: LR still decodable, go bigger.")
            continue

        print(f"  {s}x{s}: LR not decodable.")
        up_size = (s * 2, s * 2)

        bicubic = cv2.resize(lr, up_size, interpolation=cv2.INTER_CUBIC)
        espcn = espcn_upscale_bgr(lr, model, device, fp16=fp16)
        if espcn.shape[:2] != (up_size[1], up_size[0]):
            espcn = cv2.resize(espcn, up_size, interpolation=cv2.INTER_CUBIC)

        bic_tmp = tmp_dir / "bicubic.png"
        esp_tmp = tmp_dir / "espcn.png"
        cv2.imwrite(str(bic_tmp), bicubic)
        cv2.imwrite(str(esp_tmp), espcn)

        lr_ok  = is_qr_decodable(str(lr_path))
        bic_ok = is_qr_decodable(str(bic_tmp))
        esp_ok = is_qr_decodable(str(esp_tmp))

        print(f"    LR decodable:   {lr_ok}")
        print(f"    Bicubic decode: {bic_ok}")
        print(f"    ESPCN decode:   {esp_ok}")

        # Preferred strict case: LR fail, Bicubic fail, ESPCN success
        if (not lr_ok) and (not bic_ok) and esp_ok:
            gt = cv2.resize(orig, up_size, interpolation=cv2.INTER_AREA)
            chosen_s = s
            chosen_results = (gt, bicubic, espcn)
            found_strict = True
            found_any = True
            print("    -> Using STRICT sweet spot (LR fail, Bicubic fail, ESPCN success).")
            break

        # Fallback: LR fail, Bicubic & ESPCN both success (only if no strict found yet)
        if (not lr_ok) and bic_ok and esp_ok and not found_any:
            gt = cv2.resize(orig, up_size, interpolation=cv2.INTER_AREA)
            chosen_s = s
            chosen_results = (gt, bicubic, espcn)
            found_any = True
            print("    -> Using FALLBACK sweet spot candidate (LR fail, Bicubic & ESPCN success).")
            # keep looping in case a strict sweet spot appears later

    if not found_any:
        print(f"[{img_path}] No sweet spot found.")
        try:
            for p in tmp_dir.iterdir():
                p.unlink()
            tmp_dir.rmdir()
        except Exception as e:
            print(f"Warning: could not fully remove temp dir {tmp_dir}: {e}")
        return

    out_w, out_h = chosen_s * 2, chosen_s * 2
    print(f"[{img_path}] Sweet spot at LR {chosen_s}x{chosen_s}, output {out_w}x{out_h}"
          f" (strict={found_strict})")

    gt, bicubic, espcn = chosen_results
    stem = img_path.stem
    parent = img_path.parent

    gt_path  = parent / f"{stem}_GT_{out_w}x{out_h}.png"
    bic_path = parent / f"{stem}_bicubic_{out_w}x{out_h}.png"
    esp_path = parent / f"{stem}_espcn_{out_w}x{out_h}.png"

    cv2.imwrite(str(gt_path),  gt)
    cv2.imwrite(str(bic_path), bicubic)
    cv2.imwrite(str(esp_path), espcn)

    print(f"[{img_path}] Saved:")
    print("   ", gt_path.name)
    print("   ", bic_path.name)
    print("   ", esp_path.name)

    comp_path = parent / f"{stem}_comparison.png"
    compare_and_save(str(gt_path), str(bic_path), str(esp_path), save_path=str(comp_path))

    # cleanup temp dir
    try:
        for p in tmp_dir.iterdir():
            p.unlink()
        tmp_dir.rmdir()
    except Exception as e:
        print(f"Warning: could not fully remove temp dir {tmp_dir}: {e}")


# ===================== DRIVER =====================

def main() -> None:
    """
    Main entry point for QR robustness testing.
    
    Loads ESPCN model and processes all QR code images in current directory,
    finding sweet spots and generating comparison visualizations.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading ESPCN model from: {model_path}")

    model = load_espcn_pytorch(model_path, device, upscale=2)
    fp16 = False
    if fp16 and device.type == "cuda":
        model.half()

    print("Model loaded.")

    root = Path(os.getcwd())  # C:\256GB\ESPCN\results\QR
    print(f"Root: {root}")

    subfolders = [p for p in root.iterdir() if p.is_dir()]
    subfolders.sort(key=lambda p: p.name)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    for sub in subfolders:
        imgs = [p for p in sub.iterdir() if p.is_file() and p.suffix.lower() in exts]
        if not imgs:
            continue
        imgs.sort(key=lambda p: p.name)
        img_path = imgs[0]  # QR01.png etc.

        print(f"\n=== Processing folder: {sub.name}, image: {img_path.name} ===")
        find_qr_sweet_spot_for_image(
            img_path=img_path,
            model=model,
            device=device,
            min_size=8,
            step=2,
            fp16=fp16,
        )


if __name__ == "__main__":
    main()
