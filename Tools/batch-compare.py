"""
Batch Image Comparison Tool

Compares bicubic and ESPCN upscaled images against ground truth.
Calculates MSE, PSNR, and SSIM metrics, and generates comparison visualizations.

Dependencies:
    - matplotlib
    - scikit-image
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

# ============================================================================
# IMAGE COMPARISON
# ============================================================================


def compare_images(img_gt_file: str, img_bicubic_file: str, img_espcn_file: str, 
                   out_file: str) -> None:
    """Compare bicubic and ESPCN upscaling quality against ground truth.
    
    Loads three images, calculates MSE, PSNR, SSIM metrics for Bicubic and ESPCN
    against the ground truth, and saves a side-by-side comparison figure.
    
    Args:
        img_gt_file: Path to ground truth image.
        img_bicubic_file: Path to bicubic upscaled image.
        img_espcn_file: Path to ESPCN upscaled image.
        out_file: Path where comparison figure will be saved.
    """
    # --- 1. Load Images ---
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

    # --- 2. Pre-processing and Validation ---

    # Handle 4-channel PNGs (RGBA) by dropping alpha channel
    if img_gt.ndim == 3 and img_gt.shape[-1] == 4:
        img_gt = img_gt[..., :3]
    if img_bicubic.ndim == 3 and img_bicubic.shape[-1] == 4:
        img_bicubic = img_bicubic[..., :3]
    if img_espcn.ndim == 3 and img_espcn.shape[-1] == 4:
        img_espcn = img_espcn[..., :3]

    # Check if images have the same dimensions
    if not (img_gt.shape == img_bicubic.shape == img_espcn.shape):
        print("Error: Images do not have the same dimensions. Cannot compare.")
        print(f"  {img_gt_file}: {img_gt.shape}")
        print(f"  {img_bicubic_file}: {img_bicubic.shape}")
        print(f"  {img_espcn_file}: {img_espcn.shape}")
        return

    # Determine if images are grayscale or color for SSIM
    is_multichannel = img_gt.ndim == 3
    channel_axis = -1 if is_multichannel else None

    # --- 3. Calculate Metrics (only vs GT) ---

    data_range = img_gt.max() - img_gt.min()
    if data_range == 0:
        data_range = 1

    # Bicubic metrics
    mse_bicubic = mse(img_gt, img_bicubic)
    psnr_bicubic = psnr(img_gt, img_bicubic, data_range=data_range)
    ssim_bicubic = ssim(img_gt, img_bicubic, data_range=data_range,
                        channel_axis=channel_axis)

    # ESPCN metrics
    mse_espcn = mse(img_gt, img_espcn)
    psnr_espcn = psnr(img_gt, img_espcn, data_range=data_range)
    ssim_espcn = ssim(img_gt, img_espcn, data_range=data_range,
                      channel_axis=channel_axis)

    # --- 4. Print to console (optional) ---

    print("--- Image Quality Comparison ---")
    print(f"GT:       {img_gt_file}")
    print(f"Bicubic:  {img_bicubic_file}")
    print(f"ESPCN:    {img_espcn_file}")
    print()
    print(f"Bicubic -> MSE: {mse_bicubic:.2f}, PSNR: {psnr_bicubic:.2f} dB, SSIM: {ssim_bicubic:.4f}")
    print(f"ESPCN   -> MSE: {mse_espcn:.2f}, PSNR: {psnr_espcn:.2f} dB, SSIM: {ssim_espcn:.4f}")
    print("--------------------------------")

    # --- 5. Save comparison figure ---

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Ground Truth (label only, no metrics)
    axes[0].imshow(img_gt)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    # Bicubic 2x
    title_bicubic = (
        "Bicubic 2x\n"
        f"MSE: {mse_bicubic:.2f}\n"
        f"PSNR: {psnr_bicubic:.2f} dB\n"
        f"SSIM: {ssim_bicubic:.4f}"
    )
    axes[1].imshow(img_bicubic)
    axes[1].set_title(title_bicubic)
    axes[1].axis("off")

    # ESPCN 2x
    title_espcn = (
        "ESPCN 2x\n"
        f"MSE: {mse_espcn:.2f}\n"
        f"PSNR: {psnr_espcn:.2f} dB\n"
        f"SSIM: {ssim_espcn:.4f}"
    )
    axes[2].imshow(img_espcn)
    axes[2].set_title(title_espcn)
    axes[2].axis("off")

    plt.tight_layout()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close(fig)


def batch_compare(output_root: str = "./output") -> None:
    """Process batch comparison for all images.
    
    Walks ./output/<Category>/<NN>/, finds the 3 images, and writes
    <Category><NN>_comparison.png in the same folder.
    
    Args:
        output_root: Root directory containing categorized image folders.
    """
    output_root = Path(output_root)

    for category_dir in output_root.iterdir():
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        print(f"Category: {category}")

        for sample_dir in sorted(category_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            idx = sample_dir.name  # e.g. "01"
            base_name = f"{category}{idx}"

            gt_file = sample_dir / f"{base_name}.png"
            bicubic_file = sample_dir / f"{base_name}_bicubic2x.png"
            espcn_file = sample_dir / f"{base_name}_espcn2x.png"
            comp_file = sample_dir / f"{base_name}_comparison.png"

            if not (gt_file.exists() and bicubic_file.exists() and espcn_file.exists()):
                print(f"  Skipping {sample_dir}, missing one of the images.")
                continue

            print(f"  Processing {sample_dir} ...")
            compare_images(str(gt_file), str(bicubic_file),
                           str(espcn_file), str(comp_file))


if __name__ == "__main__":
    batch_compare("./addition")
