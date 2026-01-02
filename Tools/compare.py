"""
Image Quality Comparison Tool

Compares bicubic and ESPCN upscaled images against ground truth.
Displays side-by-side comparison with quality metrics (MSE, PSNR, SSIM).

Dependencies:
    - matplotlib
    - scikit-image
"""

from typing import Optional

import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

# ============================================================================
# IMAGE COMPARISON
# ============================================================================

def compare_images(img_gt_file: str, img_bicubic_file: str, img_espcn_file: str) -> None:
    """Compare bicubic and ESPCN upscaling quality against ground truth.
    
    Loads three images, calculates MSE, PSNR, and SSIM metrics,
    and displays them in a side-by-side visualization.
    
    Args:
        img_gt_file: Path to ground truth image.
        img_bicubic_file: Path to bicubic upscaled image.
        img_espcn_file: Path to ESPCN upscaled image.
    """
    
    # --- 1. Load Images ---
    try:
        img_gt = io.imread(img_gt_file)
        img_bicubic = io.imread(img_bicubic_file)
        img_espcn = io.imread(img_espcn_file)
    except FileNotFoundError as e:
        print(f"Error: Cannot find file '{e.filename}'.")
        print("Please make sure image1.png, image2.png, and image3.png are in the same directory as the script.")
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

    # --- 3. Calculate Metrics ---

    # Get the data range (e.g., 255 for 8-bit images)
    data_range = img_gt.max() - img_gt.min()

    # Calculate metrics for Bicubic (image2)
    mse_bicubic = mse(img_gt, img_bicubic)
    psnr_bicubic = psnr(img_gt, img_bicubic, data_range=data_range)
    ssim_bicubic = ssim(img_gt, img_bicubic, data_range=data_range, channel_axis=channel_axis)

    # Calculate metrics for ESPCN (image3)
    mse_espcn = mse(img_gt, img_espcn)
    psnr_espcn = psnr(img_gt, img_espcn, data_range=data_range)
    ssim_espcn = ssim(img_gt, img_espcn, data_range=data_range, channel_axis=channel_axis)

    # --- 4. Print Results to Console ---
    print("--- Image Quality Comparison ---")
    print(f"\nComparing to Ground Truth ({img_gt_file}):\n")

    print(f"Bicubic ({img_bicubic_file}):")
    print(f"  MSE:  {mse_bicubic:.2f}")
    print(f"  PSNR: {psnr_bicubic:.2f} dB")
    print(f"  SSIM: {ssim_bicubic:.4f}")

    print(f"\nESPCN ({img_espcn_file}):")
    print(f"  MSE:  {mse_espcn:.2f}")
    print(f"  PSNR: {psnr_espcn:.2f} dB")
    print(f"  SSIM: {ssim_espcn:.4f}")
    print("\n----------------------------------")
    print("Note: Lower MSE is better.")
    print("      Higher PSNR and SSIM (closer to 1.0) are better.")

    # --- 5. Display Images Side-by-Side ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Ground Truth
    axes[0].imshow(img_gt)
    axes[0].set_title(f"Ground Truth\n({img_gt_file})")
    axes[0].axis('off')

    # Bicubic
    axes[1].imshow(img_bicubic)
    title_bicubic = (
        f"Bicubic ({img_bicubic_file})\n"
        f"MSE: {mse_bicubic:.2f}\n"
        f"PSNR: {psnr_bicubic:.2f} dB\n"
        f"SSIM: {ssim_bicubic:.4f}"
    )
    axes[1].set_title(title_bicubic)
    axes[1].axis('off')

    # ESPCN
    axes[2].imshow(img_espcn)
    title_espcn = (
        f"ESPCN ({img_espcn_file})\n"
        f"MSE: {mse_espcn:.2f}\n"
        f"PSNR: {psnr_espcn:.2f} dB\n"
        f"SSIM: {ssim_espcn:.4f}"
    )
    axes[2].set_title(title_espcn)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    title_espcn = (
        f"ESPCN ({img_espcn_file})\n"
        f"MSE: {mse_espcn:.2f}\n"
        f"PSNR: {psnr_espcn:.2f} dB\n"
        f"SSIM: {ssim_espcn:.4f}"
    )
    axes[2].set_title(title_espcn)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define filenames
    IMG_GT = 'image1.png'
    IMG_BICUBIC = 'image2.png'
    IMG_ESPCN = 'image3.png'
    
    compare_images(IMG_GT, IMG_BICUBIC, IMG_ESPCN)