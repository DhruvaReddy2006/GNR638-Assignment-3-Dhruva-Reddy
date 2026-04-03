# Pix2Pix From Scratch — Assignment Guide

## Overview
This implements **"Image-to-Image Translation with Conditional Adversarial Networks"** (Isola et al., CVPR 2017) from scratch in PyTorch, and compares it against the official implementation.

**Paper:** https://arxiv.org/abs/1611.07004  
**Official Code:** https://github.com/phillipi/pix2pix

## What's Implemented (From Scratch)
- **U-Net Generator** with skip connections (Section 3.2.1)
- **70×70 PatchGAN Discriminator** (Section 3.2.2)
- **L1 + cGAN loss** with λ=100 (Equation 4)
- **Adam optimizer** with lr=0.0002, β1=0.5, β2=0.999 (Section 3.3)
- **Random jitter** (resize to 286→crop to 256) + mirroring (Appendix 6.2)
- **Weight init** from N(0, 0.02) (Appendix 6.2)
- **Facades dataset** (400 training images, quick to train)

## Quick Start (< 2 hours total)

### Step 1: Install Dependencies
```bash
pip install torch torchvision pillow matplotlib numpy
```

### Step 2: Train Our From-Scratch Model
```bash
# Full training (200 epochs, ~1-2 hours on GPU)
python pix2pix_from_scratch.py --mode train --epochs 200

# Quick test (50 epochs, ~20 min on GPU, decent results)
python pix2pix_from_scratch.py --mode train --epochs 50
```

The script auto-downloads the Facades dataset (~29MB).

### Step 3: Generate Comparison Images
```bash
python pix2pix_from_scratch.py --mode compare
```

### Step 4: Set Up Official Implementation
```bash
# Clone the PyTorch version
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
pip install -r requirements.txt

# Download facades dataset
bash ./datasets/download_pix2pix_dataset.sh facades

# Option A: Train from scratch (for fair comparison)
python train.py --dataroot ./datasets/facades --name facades_pix2pix \
    --model pix2pix --direction BtoA --n_epochs 200 --n_epochs_decay 0

# Option B: Use pretrained (faster, but trained longer)
bash ./scripts/download_pix2pix_model.sh facades_label2photo
python test.py --dataroot ./datasets/facades \
    --name facades_label2photo_pretrained --model test --direction BtoA
```

### Step 5: Run Comparison
```bash
cd ..  # back to project root
python compare_official_vs_scratch.py \
    --scratch_dir ./output_scratch/comparison \
    --official_dir ./pytorch-CycleGAN-and-pix2pix/results/facades_pix2pix/test_latest/images
```

## Output Structure
```
output_scratch/
├── images/              # Validation samples during training
│   ├── epoch_001.png
│   ├── epoch_010.png
│   └── ...
├── checkpoints/         # Model checkpoints
│   └── checkpoint_epoch_200.pth
├── comparison/          # Side-by-side comparison images
│   ├── val_000.png      # [Input | Ours | Ground Truth]
│   └── ...
├── loss_curves.png      # Training loss plots
└── training_log.txt     # CSV of losses per epoch
```

## Architecture Details

### Generator (U-Net)
```
Encoder:  C64 → C128 → C256 → C512 → C512 → C512 → C512 → C512
          (no BN on first layer, LeakyReLU 0.2 throughout)

Decoder:  CD512 → CD1024 → CD1024 → C1024 → C1024 → C512 → C256 → C128
          (50% dropout on first 3 layers, ReLU throughout)
          
Skip connections: layer i ↔ layer (n-i), concatenation
Output: ConvTranspose → Tanh
```

### Discriminator (70×70 PatchGAN)
```
C64 → C128 → C256 → C512 → 1D output
(no BN on first layer, LeakyReLU 0.2, Sigmoid output)
Input: concatenated [input_image, target/generated] = 6 channels
Output: 30×30 patch predictions
```

### Loss Function
```
G* = arg min_G max_D  L_cGAN(G, D) + λ · L_L1(G)

L_cGAN = E[log D(x,y)] + E[log(1 - D(x, G(x,z)))]
L_L1   = E[||y - G(x,z)||₁]
λ = 100
```

## Expected Results
On the Facades dataset with 200 epochs of training:
- Generated images should show recognizable building structures
- Textures (windows, walls, doors) should be plausible
- Results will be sharper than L1-only (no GAN) baseline
- Some artifacts expected, especially on complex structures

## Key Differences From Official Implementation
| Aspect | Ours | Official (PyTorch) |
|--------|------|-------------------|
| Framework | Pure PyTorch | PyTorch + custom utils |
| Dataset loading | Simple PIL + transforms | Custom aligned dataset |
| D loss function | BCE | BCE (or LSGAN option) |
| Instance norm | Not used (BN only) | Available as option |
| Noise input z | Dropout only | Dropout only |
| Learning rate schedule | Constant | Linear decay after 100 epochs |

The main functional difference is the learning rate schedule — the official version linearly decays LR to 0 over the last 100 epochs. For a fairer comparison, you can add this to the from-scratch version.

## Files
- `pix2pix_from_scratch.py` — Complete from-scratch implementation
- `compare_official_vs_scratch.py` — Comparison script and setup guide
- `README.md` — This file