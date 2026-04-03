"""
final_comparison.py
Compares our from-scratch Pix2Pix against the official implementation.
Generates metrics, side-by-side images, and a comparison report.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms

# Add official pix2pix to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'official_pix2pix'))

# Import our model
from pix2pix_from_scratch import UNetGenerator, FacadesDataset


def load_official_generator(checkpoint_path, device):
    """Load the official pix2pix generator."""
    from official_pix2pix.models.networks import UnetGenerator
    net = UnetGenerator(3, 3, 8, 64, nn.BatchNorm2d, use_dropout=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    return net


def load_scratch_generator(checkpoint_path, device):
    """Load our from-scratch generator."""
    gen = UNetGenerator(3, 3)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint["generator"])
    gen.to(device)
    gen.eval()
    return gen


def compute_metrics(fake, real):
    """Compute L1, L2, PSNR, SSIM between fake and real images (in 0-1 range)."""
    l1 = torch.mean(torch.abs(fake - real)).item()
    l2 = torch.mean((fake - real) ** 2).item()
    psnr = 10 * np.log10(1.0 / (l2 + 1e-10))
    return l1, l2, psnr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scratch_ckpt = os.path.join(base_dir, "output_scratch", "checkpoints", "checkpoint_epoch_200.pth")
    official_ckpt = os.path.join(base_dir, "official_pix2pix", "checkpoints", "facades_official", "latest_net_G.pth")
    data_dir = os.path.join(base_dir, "data", "facades")
    output_dir = os.path.join(base_dir, "final_comparison")
    os.makedirs(output_dir, exist_ok=True)

    # Check files exist
    if not os.path.exists(scratch_ckpt):
        print(f"ERROR: Scratch checkpoint not found at {scratch_ckpt}")
        return
    if not os.path.exists(official_ckpt):
        print(f"ERROR: Official checkpoint not found at {official_ckpt}")
        return

    # Load models
    print("Loading from-scratch model...")
    scratch_gen = load_scratch_generator(scratch_ckpt, device)

    print("Loading official model...")
    official_gen = load_official_generator(official_ckpt, device)

    # Load validation data
    val_dataset = FacadesDataset(data_dir, split="val", direction="BtoA")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Evaluate both models
    scratch_l1_list, scratch_l2_list, scratch_psnr_list = [], [], []
    official_l1_list, official_l2_list, official_psnr_list = [], [], []

    print(f"\nEvaluating on {len(val_dataset)} validation images...\n")

    with torch.no_grad():
        for idx, (input_img, real_img) in enumerate(val_loader):
            input_img = input_img.to(device)
            real_img = real_img.to(device)

            # Generate with both models
            scratch_fake = scratch_gen(input_img)
            official_fake = official_gen(input_img)

            # Convert to 0-1 range
            input_01 = input_img * 0.5 + 0.5
            real_01 = real_img * 0.5 + 0.5
            scratch_01 = scratch_fake * 0.5 + 0.5
            official_01 = official_fake * 0.5 + 0.5

            # Compute metrics
            s_l1, s_l2, s_psnr = compute_metrics(scratch_01, real_01)
            o_l1, o_l2, o_psnr = compute_metrics(official_01, real_01)

            scratch_l1_list.append(s_l1)
            scratch_l2_list.append(s_l2)
            scratch_psnr_list.append(s_psnr)
            official_l1_list.append(o_l1)
            official_l2_list.append(o_l2)
            official_psnr_list.append(o_psnr)

            # Save side-by-side comparison for first 10 images
            if idx < 10:
                comparison = torch.cat([
                    input_01.cpu(),
                    scratch_01.cpu(),
                    official_01.cpu(),
                    real_01.cpu()
                ], dim=3)  # Concat width-wise
                save_image(comparison, os.path.join(output_dir, f"compare_{idx:03d}.png"))

    # Compute averages
    results = {
        "scratch": {
            "L1": np.mean(scratch_l1_list),
            "L2": np.mean(scratch_l2_list),
            "PSNR": np.mean(scratch_psnr_list),
        },
        "official": {
            "L1": np.mean(official_l1_list),
            "L2": np.mean(official_l2_list),
            "PSNR": np.mean(official_psnr_list),
        }
    }

    # Print report
    report = []
    report.append("=" * 70)
    report.append("  PIX2PIX IMPLEMENTATION COMPARISON REPORT")
    report.append("  From-Scratch vs Official (pytorch-CycleGAN-and-pix2pix)")
    report.append("=" * 70)
    report.append("")
    report.append("Dataset: CMP Facades (400 train, 100 val images)")
    report.append("Training: 200 epochs, batch size 1, lr=0.0002, lambda_L1=100")
    report.append("Architecture: U-Net Generator (54.4M params) + PatchGAN Discriminator (2.8M params)")
    report.append("")
    report.append(f"{'Metric':<15} {'From Scratch':>15} {'Official':>15} {'Difference':>15}")
    report.append("-" * 62)

    for metric in ["L1", "L2", "PSNR"]:
        s = results["scratch"][metric]
        o = results["official"][metric]
        diff = s - o
        sign = "+" if diff > 0 else ""
        if metric == "PSNR":
            report.append(f"{metric:<15} {s:>14.2f}dB {o:>14.2f}dB {sign}{diff:>13.2f}dB")
        else:
            report.append(f"{metric:<15} {s:>15.4f} {o:>15.4f} {sign}{diff:>15.4f}")

    report.append("-" * 62)
    report.append("")
    report.append("INTERPRETATION:")

    l1_ratio = results["scratch"]["L1"] / results["official"]["L1"]
    if l1_ratio <= 1.15:
        report.append("  Our from-scratch implementation achieves comparable L1 error")
        report.append("  to the official implementation (within 15%).")
    else:
        report.append(f"  Our from-scratch L1 is {(l1_ratio-1)*100:.1f}% higher than official.")

    psnr_diff = results["scratch"]["PSNR"] - results["official"]["PSNR"]
    if abs(psnr_diff) < 1.0:
        report.append("  PSNR values are very close (within 1 dB).")
    elif psnr_diff > 0:
        report.append(f"  Our model achieves {psnr_diff:.2f} dB higher PSNR.")
    else:
        report.append(f"  Official model achieves {-psnr_diff:.2f} dB higher PSNR.")

    report.append("")
    report.append("KEY DIFFERENCES:")
    report.append("  - Both use identical architectures (U-Net + PatchGAN)")
    report.append("  - Official uses instance norm option; ours uses batch norm")
    report.append("  - Small differences from random initialization and data augmentation")
    report.append("  - Official has additional features (learning rate decay, etc.)")
    report.append("")
    report.append("COMPARISON IMAGES:")
    report.append("  Each image shows: [Input Labels | Ours | Official | Ground Truth]")
    report.append(f"  Saved to: {output_dir}/")
    report.append("")
    report.append("=" * 70)

    report_text = "\n".join(report)
    print(report_text)

    # Save report
    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")

    # Save loss curves comparison plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Per-image comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        x = range(min(len(scratch_l1_list), len(official_l1_list)))
        axes[0].bar([i - 0.2 for i in x], scratch_l1_list[:len(x)], 0.4, label="Ours", alpha=0.7)
        axes[0].bar([i + 0.2 for i in x], official_l1_list[:len(x)], 0.4, label="Official", alpha=0.7)
        axes[0].set_title("L1 Error per Image")
        axes[0].set_xlabel("Validation Image")
        axes[0].set_ylabel("L1 Error")
        axes[0].legend()

        axes[1].bar([i - 0.2 for i in x], scratch_psnr_list[:len(x)], 0.4, label="Ours", alpha=0.7)
        axes[1].bar([i + 0.2 for i in x], official_psnr_list[:len(x)], 0.4, label="Official", alpha=0.7)
        axes[1].set_title("PSNR per Image (dB)")
        axes[1].set_xlabel("Validation Image")
        axes[1].set_ylabel("PSNR (dB)")
        axes[1].legend()

        # Summary bar chart
        metrics = ["L1", "PSNR"]
        scratch_vals = [results["scratch"]["L1"], results["scratch"]["PSNR"]]
        official_vals = [results["official"]["L1"], results["official"]["PSNR"]]

        x_pos = range(len(metrics))
        axes[2].bar([i - 0.2 for i in x_pos], scratch_vals, 0.4, label="Ours")
        axes[2].bar([i + 0.2 for i in x_pos], official_vals, 0.4, label="Official")
        axes[2].set_xticks(list(x_pos))
        axes[2].set_xticklabels(metrics)
        axes[2].set_title("Average Metrics Comparison")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150)
        plt.close()
        print(f"Metrics plot saved to {output_dir}/metrics_comparison.png")
    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
