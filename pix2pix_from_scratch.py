"""
Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks
From-scratch implementation following Isola et al. (CVPR 2017)

This implements:
- U-Net Generator with skip connections
- PatchGAN Discriminator (70x70 receptive field)
- Combined L1 + cGAN loss (Equation 4 in the paper)
- Training on the Facades dataset (400 images, quick to train)

Usage:
    python pix2pix_from_scratch.py --mode train
    python pix2pix_from_scratch.py --mode compare
"""

import os
import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import urllib.request
import tarfile


# =============================================================================
# 1. DATASET
# =============================================================================

class FacadesDataset(Dataset):
    """
    Facades dataset: Each image is a 256x512 side-by-side pair.
    Left half = real photo, Right half = label map (input).
    We want to translate labels -> photo.
    """
    def __init__(self, root_dir, split="train", direction="BtoA"):
        self.root_dir = os.path.join(root_dir, split)
        self.image_files = sorted([
            f for f in os.listdir(self.root_dir)
            if f.endswith(('.jpg', '.png'))
        ])
        self.direction = direction  # BtoA = labels->photo

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        # Split the 512x256 image into two 256x256 halves
        w, h = img.size
        half_w = w // 2

        # In the facades dataset: left=real photo (A), right=label map (B)
        img_A = img.crop((0, 0, half_w, h))       # Real photo
        img_B = img.crop((half_w, 0, w, h))        # Label map

        # Random jitter: resize to 286x286, then random crop to 256x256
        if self.root_dir.endswith("train"):
            resize = transforms.Resize((286, 286))
            img_A = resize(img_A)
            img_B = resize(img_B)

            # Random crop
            i, j, th, tw = transforms.RandomCrop.get_params(
                img_A, output_size=(256, 256)
            )
            img_A = transforms.functional.crop(img_A, i, j, th, tw)
            img_B = transforms.functional.crop(img_B, i, j, th, tw)

            # Random horizontal flip
            if np.random.random() > 0.5:
                img_A = transforms.functional.hflip(img_A)
                img_B = transforms.functional.hflip(img_B)
        else:
            resize = transforms.Resize((256, 256))
            img_A = resize(img_A)
            img_B = resize(img_B)

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        if self.direction == "BtoA":
            return img_B, img_A  # input=labels, target=photo
        else:
            return img_A, img_B


def download_facades(data_dir="./data/facades"):
    """Download the facades dataset."""
    if os.path.exists(os.path.join(data_dir, "train")):
        print("Facades dataset already downloaded.")
        return data_dir

    os.makedirs(data_dir, exist_ok=True)
    url = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
    tar_path = os.path.join(data_dir, "facades.tar.gz")

    print(f"Downloading facades dataset from {url}...")
    urllib.request.urlretrieve(url, tar_path)

    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=os.path.dirname(data_dir))

    os.remove(tar_path)
    print("Done!")
    return data_dir


# =============================================================================
# 2. U-NET GENERATOR (Section 3.2.1 of the paper)
# =============================================================================

class UNetDown(nn.Module):
    """Encoder block: Conv -> BatchNorm -> LeakyReLU"""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Decoder block: ConvTranspose -> BatchNorm -> (Dropout) -> ReLU"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        # Skip connection: concatenate encoder features
        x = torch.cat((x, skip_input), dim=1)
        return x


class UNetGenerator(nn.Module):
    """
    U-Net Generator as described in Section 3.2.1 and Appendix 6.1.1.

    Architecture (for 256x256 input):
    Encoder: C64 - C128 - C256 - C512 - C512 - C512 - C512 - C512
    Decoder: CD512 - CD1024 - CD1024 - C1024 - C1024 - C512 - C256 - C128

    Skip connections between layer i and layer n-i.
    No BatchNorm on first encoder layer.
    Dropout (50%) on first 3 decoder layers.
    Encoder uses LeakyReLU(0.2), decoder uses ReLU.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Encoder (downsampling)
        self.down1 = UNetDown(in_channels, 64, normalize=False)   # 128x128
        self.down2 = UNetDown(64, 128)                             # 64x64
        self.down3 = UNetDown(128, 256)                            # 32x32
        self.down4 = UNetDown(256, 512)                            # 16x16
        self.down5 = UNetDown(512, 512)                            # 8x8
        self.down6 = UNetDown(512, 512)                            # 4x4
        self.down7 = UNetDown(512, 512)                            # 2x2
        self.down8 = UNetDown(512, 512, normalize=False)           # 1x1

        # Decoder (upsampling) - channels doubled due to skip connections
        self.up1 = UNetUp(512, 512, dropout=0.5)     # 2x2,   out=1024 (512+512 skip)
        self.up2 = UNetUp(1024, 512, dropout=0.5)    # 4x4,   out=1024
        self.up3 = UNetUp(1024, 512, dropout=0.5)    # 8x8,   out=1024
        self.up4 = UNetUp(1024, 512)                  # 16x16, out=1024
        self.up5 = UNetUp(1024, 256)                  # 32x32, out=512
        self.up6 = UNetUp(512, 128)                   # 64x64, out=256
        self.up7 = UNetUp(256, 64)                    # 128x128, out=128

        # Final layer: upsample to 256x256 and map to output channels
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


# =============================================================================
# 3. PATCHGAN DISCRIMINATOR (Section 3.2.2)
# =============================================================================

class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator as described in Section 3.2.2 and Appendix 6.1.2.

    Architecture: C64 - C128 - C256 - C512 -> 1D output
    No BatchNorm on first layer. All ReLUs are leaky (0.2).

    Input: concatenation of input image and target/generated image (6 channels).
    Output: 30x30 patch of real/fake predictions.
    """
    def __init__(self, in_channels=6):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # C64 - no batchnorm
            *discriminator_block(in_channels, 64, normalize=False),
            # C128
            *discriminator_block(64, 128),
            # C256
            *discriminator_block(128, 256),
            # C512 - stride 1 for last conv block before output
            nn.ZeroPad2d((1, 0, 1, 0)),  # Pad to maintain size
            nn.Conv2d(256, 512, 4, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output 1-channel prediction map
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img_input, img_target):
        # Concatenate input and target along channel dimension
        # This is the "conditional" part - D sees both input and output
        x = torch.cat((img_input, img_target), dim=1)
        return self.model(x)


# =============================================================================
# 4. WEIGHT INITIALIZATION (Appendix 6.2)
# =============================================================================

def init_weights(m):
    """Initialize weights from N(0, 0.02) as specified in the paper."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# =============================================================================
# 5. TRAINING LOOP
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download and load dataset
    data_dir = download_facades(args.data_dir)
    train_dataset = FacadesDataset(data_dir, split="train", direction="BtoA")
    val_dataset = FacadesDataset(data_dir, split="val", direction="BtoA")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)

    # Initialize models
    generator = UNetGenerator(in_channels=3, out_channels=3).to(device)
    discriminator = PatchGANDiscriminator(in_channels=6).to(device)

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    # Losses
    criterion_GAN = nn.BCELoss()  # Adversarial loss
    criterion_L1 = nn.L1Loss()    # L1 reconstruction loss

    # Optimizers - Adam with lr=0.0002, beta1=0.5, beta2=0.999 (Section 3.3)
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Lambda for L1 loss (lambda=100 as in the paper, Equation 4)
    lambda_L1 = 100.0

    # Output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Dataset size: {len(train_dataset)} training images")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Lambda L1: {lambda_L1}")
    print(f"  Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"  Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
    print()

    train_losses = {"G": [], "D": [], "L1": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        epoch_G_loss = 0
        epoch_D_loss = 0
        epoch_L1_loss = 0

        for batch_idx, (input_img, real_img) in enumerate(train_loader):
            input_img = input_img.to(device)
            real_img = real_img.to(device)

            # --------------------------------------------------
            # Train Discriminator
            # --------------------------------------------------
            optimizer_D.zero_grad()

            # Real pair
            pred_real = discriminator(input_img, real_img)
            real_label = torch.ones_like(pred_real, device=device)
            fake_label = torch.zeros_like(pred_real, device=device)
            loss_D_real = criterion_GAN(pred_real, real_label)

            # Fake pair
            fake_img = generator(input_img)
            pred_fake = discriminator(input_img, fake_img.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake_label)

            # Total D loss (divided by 2 as mentioned in Section 3.3)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # --------------------------------------------------
            # Train Generator
            # --------------------------------------------------
            optimizer_G.zero_grad()

            # GAN loss: G wants D to think fakes are real
            pred_fake_for_G = discriminator(input_img, fake_img)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G, device=device))

            # L1 loss
            loss_G_L1 = criterion_L1(fake_img, real_img)

            # Total G loss: L_cGAN + lambda * L_L1  (Equation 4)
            loss_G = loss_G_GAN + lambda_L1 * loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            epoch_G_loss += loss_G.item()
            epoch_D_loss += loss_D.item()
            epoch_L1_loss += loss_G_L1.item()

        # Average losses
        n_batches = len(train_loader)
        avg_G = epoch_G_loss / n_batches
        avg_D = epoch_D_loss / n_batches
        avg_L1 = epoch_L1_loss / n_batches

        train_losses["G"].append(avg_G)
        train_losses["D"].append(avg_D)
        train_losses["L1"].append(avg_L1)

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch}/{args.epochs}] | "
              f"D_loss: {avg_D:.4f} | G_loss: {avg_G:.4f} | L1: {avg_L1:.4f} | "
              f"Time: {elapsed:.0f}s")

        # Save sample images every N epochs
        if epoch % args.save_interval == 0 or epoch == 1:
            save_val_samples(generator, val_loader, device, epoch, args.output_dir)

        # Save checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            torch.save({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "losses": train_losses,
            }, os.path.join(args.output_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pth"))

    print(f"\nTraining complete! Total time: {time.time() - start_time:.0f}s")
    print(f"Results saved to {args.output_dir}/")

    # Save final loss log
    save_loss_plot(train_losses, args.output_dir)

    return generator, train_losses


def save_val_samples(generator, val_loader, device, epoch, output_dir):
    """Save a grid of validation results."""
    generator.eval()
    with torch.no_grad():
        for input_img, real_img in val_loader:
            input_img = input_img.to(device)
            fake_img = generator(input_img)

            # Denormalize
            input_img = input_img * 0.5 + 0.5
            real_img = real_img * 0.5 + 0.5
            fake_img = fake_img * 0.5 + 0.5

            # Save side by side: input | generated | ground truth
            comparison = torch.cat([
                input_img.cpu()[:4],
                fake_img.cpu()[:4],
                real_img[:4]
            ], dim=0)
            save_image(
                comparison,
                os.path.join(output_dir, "images", f"epoch_{epoch:03d}.png"),
                nrow=4, padding=2
            )
            break
    generator.train()


def save_loss_plot(losses, output_dir):
    """Save loss curves as a simple text log + matplotlib plot if available."""
    log_path = os.path.join(output_dir, "training_log.txt")
    with open(log_path, "w") as f:
        f.write("epoch,G_loss,D_loss,L1_loss\n")
        for i in range(len(losses["G"])):
            f.write(f"{i+1},{losses['G'][i]:.6f},{losses['D'][i]:.6f},{losses['L1'][i]:.6f}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(losses["G"], label="Generator")
        ax1.plot(losses["D"], label="Discriminator")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("GAN Losses")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(losses["L1"], label="L1 Loss", color="green")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("L1 Loss")
        ax2.set_title("Reconstruction Loss (L1)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
        plt.close()
        print(f"Loss curves saved to {output_dir}/loss_curves.png")
    except ImportError:
        print("matplotlib not available, skipping loss plot (text log saved)")


# =============================================================================
# 6. EVALUATION / COMPARISON
# =============================================================================

def evaluate_model(generator, data_loader, device):
    """Compute L1 and L2 metrics on validation set."""
    generator.eval()
    total_l1 = 0
    total_l2 = 0
    total_psnr = 0
    count = 0

    with torch.no_grad():
        for input_img, real_img in data_loader:
            input_img = input_img.to(device)
            real_img = real_img.to(device)
            fake_img = generator(input_img)

            # Compute metrics in [0, 1] range
            fake_01 = fake_img * 0.5 + 0.5
            real_01 = real_img * 0.5 + 0.5

            l1 = torch.mean(torch.abs(fake_01 - real_01)).item()
            l2 = torch.mean((fake_01 - real_01) ** 2).item()
            psnr = 10 * np.log10(1.0 / (l2 + 1e-10))

            total_l1 += l1
            total_l2 += l2
            total_psnr += psnr
            count += 1

    return {
        "L1": total_l1 / count,
        "L2": total_l2 / count,
        "PSNR": total_psnr / count,
    }


def compare_implementations(args):
    """
    Compare our from-scratch implementation against the official pix2pix.
    Generates a side-by-side comparison report.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = download_facades(args.data_dir)
    val_dataset = FacadesDataset(data_dir, split="val", direction="BtoA")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Load our trained model
    ckpt_path = os.path.join(args.output_dir, "checkpoints",
                             f"checkpoint_epoch_{args.epochs}.pth")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        print("Please train first: python pix2pix_from_scratch.py --mode train")
        return

    generator = UNetGenerator().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])

    # Evaluate
    metrics = evaluate_model(generator, val_loader, device)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS (Our From-Scratch Implementation)")
    print("=" * 60)
    print(f"  Mean L1 Error:    {metrics['L1']:.4f}")
    print(f"  Mean L2 Error:    {metrics['L2']:.6f}")
    print(f"  Mean PSNR:        {metrics['PSNR']:.2f} dB")
    print("=" * 60)

    # Save comparison images
    generator.eval()
    os.makedirs(os.path.join(args.output_dir, "comparison"), exist_ok=True)

    with torch.no_grad():
        for idx, (input_img, real_img) in enumerate(val_loader):
            if idx >= 10:
                break
            input_img = input_img.to(device)
            fake_img = generator(input_img)

            # Denormalize
            imgs = torch.cat([
                input_img.cpu() * 0.5 + 0.5,
                fake_img.cpu() * 0.5 + 0.5,
                real_img * 0.5 + 0.5,
            ], dim=3)  # Concat width-wise

            save_image(
                imgs,
                os.path.join(args.output_dir, "comparison", f"val_{idx:03d}.png")
            )

    print(f"\nComparison images saved to {args.output_dir}/comparison/")
    print("Each image shows: [Input Labels] | [Our Output] | [Ground Truth]")
    print("\nTo compare against official implementation:")
    print("  1. Clone: git clone https://github.com/phillipi/pix2pix")
    print("  2. Follow their instructions to train on facades")
    print("  3. Compare the output images side-by-side")
    print("  4. Run their test script and compare FCN-scores if desired")


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pix2Pix From Scratch")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "compare"],
                        help="train or compare")
    parser.add_argument("--data_dir", type=str, default="./data/facades",
                        help="Path to facades dataset")
    parser.add_argument("--output_dir", type=str, default="./output_scratch",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (paper uses 1 for facades)")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save images/checkpoints every N epochs")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "compare":
        compare_implementations(args)


if __name__ == "__main__":
    main()