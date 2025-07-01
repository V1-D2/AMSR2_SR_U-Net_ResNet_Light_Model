#!/usr/bin/env python3
"""
AMSR2 Sequential Trainer - GPU Optimized Edition
Sequential file processing with GPU optimizations and mixed precision support

Author: Volodymyr Didur
Version: 5.0 - GPU Optimized Sequential Processing
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import time
import json
import argparse
from pathlib import Path
import psutil
import sys
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# GPU-optimized thread settings
if torch.cuda.is_available():
    # For GPU, limit CPU threads to avoid contention
    torch.set_num_threads(4)
    # Enable cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
else:
    # For CPU, use more threads
    torch.set_num_threads(min(8, os.cpu_count()))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('amsr2_gpu_sequential.log')
    ]
)
logger = logging.getLogger(__name__)


# ====== DATASET FOR SINGLE FILE (Keep most of the original) ======
class SingleFileAMSR2Dataset(Dataset):
    """Dataset for processing one NPZ file"""

    def __init__(self, npz_path: str, preprocessor,
                 degradation_scale: int = 4, augment: bool = True,
                 filter_orbit_type: Optional[str] = None):

        self.npz_path = npz_path
        self.preprocessor = preprocessor
        self.degradation_scale = degradation_scale
        self.augment = augment
        self.filter_orbit_type = filter_orbit_type

        # Load data from file
        self.swaths = self._load_file_data()

    def _load_file_data(self):
        """Load data from a single file"""
        logger.info(f"📂 Loading file: {os.path.basename(self.npz_path)}")

        try:
            with np.load(self.npz_path, allow_pickle=True) as data:
                if 'swath_array' not in data:
                    logger.error(f"❌ Invalid file structure: {self.npz_path}")
                    return []

                swath_array = data['swath_array']
                valid_swaths = []

                for swath_idx, swath_dict in enumerate(swath_array):
                    swath = swath_dict.item() if isinstance(swath_dict, np.ndarray) else swath_dict

                    # Check structure
                    if 'temperature' not in swath or 'metadata' not in swath:
                        logger.warning(f"⚠️ Skipping swath {swath_idx}: invalid structure")
                        continue

                    # Filter by orbit type
                    if self.filter_orbit_type is not None:
                        orbit_type = swath['metadata'].get('orbit_type', 'U')
                        if orbit_type != self.filter_orbit_type:
                            continue

                    # Process temperature
                    raw_temperature = swath['temperature']
                    metadata = swath['metadata']

                    # Apply scale factor
                    scale_factor = metadata.get('scale_factor', 1.0)

                    if raw_temperature.dtype != np.float32:
                        temperature = raw_temperature.astype(np.float32) * scale_factor
                    else:
                        temperature = raw_temperature * scale_factor

                    # Filter invalid values
                    temperature = np.where(temperature < 50, np.nan, temperature)
                    temperature = np.where(temperature > 350, np.nan, temperature)

                    # Check validity
                    valid_pixels = np.sum(~np.isnan(temperature))
                    total_pixels = temperature.size

                    if valid_pixels / total_pixels < 0.1:
                        continue

                    valid_swaths.append({
                        'temperature': temperature,
                        'metadata': metadata
                    })

                logger.info(f"✅ Loaded {len(valid_swaths)} valid swaths out of {len(swath_array)}")
                return valid_swaths

        except Exception as e:
            logger.error(f"❌ Error loading file {self.npz_path}: {e}")
            return []

    def __len__(self):
        return len(self.swaths)

    def __getitem__(self, idx):
        if idx >= len(self.swaths):
            empty_tensor = torch.zeros(1, self.preprocessor.target_height, self.preprocessor.target_width)
            return empty_tensor, empty_tensor

        try:
            swath = self.swaths[idx]
            temperature = swath['temperature'].copy()

            # Process data
            temperature = self.preprocessor.crop_and_pad_to_target(temperature)
            temperature = self.preprocessor.normalize_brightness_temperature(temperature)

            if self.augment:
                temperature = self._augment_data(temperature)

            degraded = self._create_degradation(temperature)

            high_res = torch.from_numpy(temperature).unsqueeze(0).float()
            low_res = torch.from_numpy(degraded).unsqueeze(0).float()

            return low_res, high_res

        except Exception as e:
            logger.error(f"❌ Error in __getitem__[{idx}]: {e}")
            empty_tensor = torch.zeros(1, self.preprocessor.target_height, self.preprocessor.target_width)
            return empty_tensor, empty_tensor

    def _create_degradation(self, high_res: np.ndarray) -> np.ndarray:
        """Create degraded version using PyTorch operations"""
        h, w = high_res.shape

        # Use PyTorch for degradation (more efficient on GPU)
        high_res_tensor = torch.from_numpy(high_res).unsqueeze(0).unsqueeze(0)

        # Downsample
        low_res = F.interpolate(
            high_res_tensor,
            size=(h // self.degradation_scale, w // self.degradation_scale),
            mode='bilinear',
            align_corners=False
        )

        # Add noise
        noise = torch.randn_like(low_res) * 0.01
        low_res = low_res + noise

        # Blur
        blur = transforms.GaussianBlur(kernel_size=3, sigma=0.3)
        low_res = blur(low_res)

        # Upsample back
        degraded = F.interpolate(
            low_res,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )

        return degraded.squeeze().numpy()

    def _augment_data(self, data: np.ndarray) -> np.ndarray:
        """Simple augmentation"""
        if not self.augment or np.random.rand() > 0.3:
            return data

        if np.random.rand() > 0.5:
            data = np.fliplr(data)
        if np.random.rand() > 0.5:
            data = np.flipud(data)

        return data


# ====== PREPROCESSOR (Keep as is) ======
class AMSR2NPZDataPreprocessor:
    """Preprocessor for AMSR2 data"""

    def __init__(self, target_height: int = 2000, target_width: int = 420):
        self.target_height = target_height
        self.target_width = target_width
        logger.info(f"📏 Preprocessor configured for size: {target_height}x{target_width}")

    def crop_and_pad_to_target(self, temperature: np.ndarray) -> np.ndarray:
        """Crop or pad to target size"""
        original_shape = temperature.shape
        h, w = original_shape

        # Crop if larger
        if h > self.target_height:
            start_h = (h - self.target_height) // 2
            temperature = temperature[start_h:start_h + self.target_height]

        if w > self.target_width:
            start_w = (w - self.target_width) // 2
            temperature = temperature[:, start_w:start_w + self.target_width]

        current_h, current_w = temperature.shape

        # Pad if smaller
        if current_h < self.target_height or current_w < self.target_width:
            pad_h = max(0, self.target_height - current_h)
            pad_w = max(0, self.target_width - current_w)

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            temperature = np.pad(temperature,
                                 ((pad_top, pad_bottom), (pad_left, pad_right)),
                                 mode='reflect')

        final_shape = temperature.shape
        if original_shape != final_shape:
            logger.debug(f"Size changed: {original_shape} → {final_shape}")

        return temperature

    def normalize_brightness_temperature(self, temperature: np.ndarray) -> np.ndarray:
        """Normalize brightness temperature"""
        valid_mask = ~np.isnan(temperature)
        if np.sum(valid_mask) > 0:
            mean_temp = np.mean(temperature[valid_mask])
            temperature = np.where(np.isnan(temperature), mean_temp, temperature)
        else:
            temperature = np.full_like(temperature, 250.0)

        temperature = np.clip(temperature, 50, 350)
        normalized = (temperature - 200) / 150

        return normalized.astype(np.float32)


# ====== MODEL ARCHITECTURE (Keep as is) ======
# [ResNetBlock, UNetResNetEncoder, UNetDecoder, UNetResNetSuperResolution classes remain the same]
# I'll include just the class definitions to save space

class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class UNetResNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        x = F.relu(self.bn1(self.conv1(x)))
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return x, features


class UNetDecoder(nn.Module):
    def __init__(self, out_channels: int = 1):
        super().__init__()
        self.up4 = self._make_upconv_block(512, 256)
        self.up3 = self._make_upconv_block(256 + 256, 128)
        self.up2 = self._make_upconv_block(128 + 128, 64)
        self.up1 = self._make_upconv_block(64 + 64, 64)
        self.final_up = nn.ConvTranspose2d(64 + 64, 32, 2, 2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 1)
        )

    def _make_upconv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_features):
        x = self.up4(x)
        x = torch.cat([x, skip_features[3]], dim=1)
        x = self.up3(x)
        x = torch.cat([x, skip_features[2]], dim=1)
        x = self.up2(x)
        x = torch.cat([x, skip_features[1]], dim=1)
        x = self.up1(x)
        x = torch.cat([x, skip_features[0]], dim=1)
        x = self.final_up(x)
        x = self.final_conv(x)
        return x


class UNetResNetSuperResolution(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, scale_factor: int = 10):
        super().__init__()
        self.scale_factor = scale_factor
        self.encoder = UNetResNetEncoder(in_channels)
        self.decoder = UNetDecoder(out_channels)

        if scale_factor > 1:
            upsampling_layers = []
            current_scale = 1

            while current_scale < scale_factor:
                if scale_factor // current_scale >= 4:
                    factor = 4
                elif scale_factor // current_scale >= 2:
                    factor = 2
                else:
                    factor = scale_factor // current_scale

                upsampling_layers.extend([
                    nn.ConvTranspose2d(out_channels if len(upsampling_layers) == 0 else 32, 32,
                                       factor, factor),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.ReLU(inplace=True)
                ])
                current_scale *= factor

            upsampling_layers.append(nn.Conv2d(32, out_channels, 1))
            self.upsampling = nn.Sequential(*upsampling_layers)
        else:
            self.upsampling = nn.Identity()

    def forward(self, x):
        encoded, skip_features = self.encoder(x)
        decoded = self.decoder(encoded, skip_features)
        output = self.upsampling(decoded)
        return output


# ====== LOSS FUNCTION (Keep as is) ======
class AMSR2SpecificLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.15, gamma: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def compute_gradients(x):
            grad_x = x[:, :, :-1, :] - x[:, :, 1:, :]
            grad_y = x[:, :, :, :-1] - x[:, :, :, 1:]
            return grad_x, grad_y

        pred_grad_x, pred_grad_y = compute_gradients(pred)
        target_grad_x, target_grad_y = compute_gradients(target)

        loss_x = self.l1_loss(pred_grad_x, target_grad_x)
        loss_y = self.l1_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y

    def brightness_temperature_consistency(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = torch.mean(pred, dim=[2, 3])
        target_mean = torch.mean(target, dim=[2, 3])
        energy_loss = self.mse_loss(pred_mean, target_mean)

        pred_std = torch.std(pred, dim=[2, 3])
        target_std = torch.std(target, dim=[2, 3])
        distribution_loss = self.mse_loss(pred_std, target_std)

        range_penalty = torch.mean(torch.relu(torch.abs(pred) - 1.0))

        return energy_loss + 0.5 * distribution_loss + 0.1 * range_penalty

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        l1_loss = self.l1_loss(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        phys_loss = self.brightness_temperature_consistency(pred, target)

        total_loss = (self.alpha * l1_loss +
                      self.beta * grad_loss +
                      self.gamma * phys_loss)

        return total_loss, {
            'l1_loss': l1_loss.item(),
            'gradient_loss': grad_loss.item(),
            'physical_loss': phys_loss.item(),
            'total_loss': total_loss.item()
        }


# ====== GPU-OPTIMIZED SEQUENTIAL TRAINER ======
class GPUSequentialAMSR2Trainer:
    """GPU-optimized sequential trainer with mixed precision support"""

    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5,
                 use_amp: bool = True):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # Mixed precision scaler for GPU
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.criterion = AMSR2SpecificLoss()
        self.training_history = []
        self.best_loss = float('inf')

    def train_on_file(self, file_path: str, preprocessor, batch_size: int = 4,
                      augment: bool = True, filter_orbit_type: Optional[str] = None):
        """Train on one file with GPU optimizations"""

        logger.info(f"📚 Training on file: {os.path.basename(file_path)}")

        # Clear GPU cache before starting
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Create dataset for one file
        dataset = SingleFileAMSR2Dataset(
            npz_path=file_path,
            preprocessor=preprocessor,
            degradation_scale=4,
            augment=augment,
            filter_orbit_type=filter_orbit_type
        )

        if len(dataset) == 0:
            logger.warning(f"⚠️ Empty file, skipping: {file_path}")
            return {'loss': float('inf'), 'swaths': 0}

        # DataLoader with GPU optimizations
        num_workers = 4 if self.device.type == 'cuda' else 0
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )

        self.model.train()
        file_losses = []

        try:
            progress_bar = tqdm(dataloader, desc="Batches", leave=False)
            for batch_idx, (low_res, high_res) in enumerate(progress_bar):
                low_res = low_res.to(self.device)
                high_res = high_res.to(self.device)

                self.optimizer.zero_grad()

                # Mixed precision training
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    pred = self.model(low_res)
                    loss, loss_components = self.criterion(pred, high_res)

                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                file_losses.append(loss.item())

                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

                # Clear cache periodically on GPU
                if batch_idx % 10 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("💾 GPU out of memory! Clearing cache and skipping file")
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                return {'loss': float('inf'), 'swaths': 0}
            else:
                raise e

        except Exception as e:
            logger.error(f"❌ Error training on file {file_path}: {e}")
            return {'loss': float('inf'), 'swaths': 0}

        finally:
            # Clean up
            del dataset, dataloader
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        avg_loss = np.mean(file_losses) if file_losses else float('inf')
        logger.info(f"✅ File processed: loss={avg_loss:.4f}, batches={len(file_losses)}")

        return {'loss': avg_loss, 'swaths': len(dataset) if 'dataset' in locals() else 0}

    def train_sequential(self, npz_files: List[str], preprocessor,
                         epochs_per_file: int = 1, batch_size: int = 4,
                         save_path: str = "best_amsr2_model.pth"):
        """Sequential training on all files"""

        logger.info(f"🚀 Starting GPU sequential training:")
        logger.info(f"   Files: {len(npz_files)}")
        logger.info(f"   Epochs per file: {epochs_per_file}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"   Mixed precision: {self.use_amp}")
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        total_files = len(npz_files)
        processed_files = 0

        for file_idx, file_path in enumerate(npz_files):
            logger.info(f"\n📂 File {file_idx + 1}/{total_files}: {os.path.basename(file_path)}")

            file_results = []

            # Multiple epochs on one file
            for epoch in range(epochs_per_file):
                logger.info(f"   Epoch {epoch + 1}/{epochs_per_file}")

                result = self.train_on_file(
                    file_path=file_path,
                    preprocessor=preprocessor,
                    batch_size=batch_size,
                    augment=True,
                    filter_orbit_type=None
                )

                file_results.append(result)

                # Save best model
                if result['loss'] < self.best_loss and result['loss'] != float('inf'):
                    self.best_loss = result['loss']
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.best_loss,
                        'file_idx': file_idx,
                        'epoch': epoch,
                        'device': str(self.device)
                    }, save_path)
                    logger.info(f"💾 Saved best model: loss={self.best_loss:.4f}")

            # File statistics
            valid_losses = [r['loss'] for r in file_results if r['loss'] != float('inf')]
            avg_file_loss = np.mean(valid_losses) if valid_losses else float('inf')
            total_swaths = sum(r['swaths'] for r in file_results)

            self.training_history.append({
                'file_idx': file_idx,
                'file_path': file_path,
                'avg_loss': avg_file_loss,
                'total_swaths': total_swaths,
                'epochs': epochs_per_file
            })

            processed_files += 1

            logger.info(f"📊 File completed: avg_loss={avg_file_loss:.4f}, swaths={total_swaths}")
            logger.info(f"   Progress: {processed_files}/{total_files} files")

            # Scheduler step
            if avg_file_loss != float('inf'):
                self.scheduler.step(avg_file_loss)

            # GPU memory logging
            if self.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024 ** 3
                reserved = torch.cuda.memory_reserved() / 1024 ** 3
                logger.info(f"   GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

        logger.info(f"\n🎉 Sequential training completed!")
        logger.info(f"   Files processed: {processed_files}/{total_files}")
        logger.info(f"   Best loss: {self.best_loss:.4f}")

        return self.training_history


# ====== UTILITY FUNCTIONS ======
def find_npz_files(directory: str, max_files: Optional[int] = None) -> List[str]:
    """Find NPZ files with optional limit"""

    if not os.path.exists(directory):
        logger.error(f"❌ Directory does not exist: {directory}")
        return []

    pattern = os.path.join(directory, "*.npz")
    all_files = glob.glob(pattern)

    if not all_files:
        logger.error(f"❌ No NPZ files found in directory: {directory}")
        return []

    # Sort files for reproducibility
    all_files.sort()

    if max_files is not None and max_files > 0:
        selected_files = all_files[:max_files]
        logger.info(f"📁 Found {len(all_files)} NPZ files, selected {len(selected_files)}")
    else:
        selected_files = all_files
        logger.info(f"📁 Found {len(selected_files)} NPZ files")

    # Check file sizes
    total_size_gb = 0
    for file_path in selected_files:
        size_gb = os.path.getsize(file_path) / 1024 ** 3
        total_size_gb += size_gb

    logger.info(f"📊 Total data size: {total_size_gb:.2f} GB")

    return selected_files


def create_training_summary(training_history: List[Dict], save_path: str = "training_summary_gpu.json"):
    """Create training summary"""

    if not training_history:
        return

    valid_results = [h for h in training_history if h['avg_loss'] != float('inf')]

    if not valid_results:
        logger.warning("⚠️ No valid results for summary")
        return

    summary = {
        'total_files_processed': len(training_history),
        'successful_files': len(valid_results),
        'total_swaths': sum(h['total_swaths'] for h in valid_results),
        'best_loss': min(h['avg_loss'] for h in valid_results),
        'worst_loss': max(h['avg_loss'] for h in valid_results),
        'average_loss': np.mean([h['avg_loss'] for h in valid_results]),
        'training_history': training_history,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"📄 Summary saved: {save_path}")
    logger.info(f"   Successful files: {summary['successful_files']}/{summary['total_files_processed']}")
    logger.info(f"   Total swaths: {summary['total_swaths']}")
    logger.info(f"   Best loss: {summary['best_loss']:.4f}")


def plot_training_progress(training_history: List[Dict], save_path: str = "training_progress_gpu.png"):
    """Visualize training progress"""

    if not training_history:
        return

    valid_results = [h for h in training_history if h['avg_loss'] != float('inf')]

    if len(valid_results) < 2:
        logger.warning("⚠️ Not enough data for plot")
        return

    file_indices = [h['file_idx'] for h in valid_results]
    losses = [h['avg_loss'] for h in valid_results]
    swaths = [h['total_swaths'] for h in valid_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Loss plot
    ax1.plot(file_indices, losses, 'b-o', linewidth=2, markersize=6)
    ax1.set_title('Training Loss by File', fontsize=14)
    ax1.set_xlabel('File Index')
    ax1.set_ylabel('Average Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Swaths plot
    ax2.bar(file_indices, swaths, alpha=0.7, color='green')
    ax2.set_title('Swaths per File', fontsize=14)
    ax2.set_xlabel('File Index')
    ax2.set_ylabel('Number of Swaths')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"📈 Plot saved: {save_path}")


# ====== MAIN FUNCTION ======
def main():
    """Main function for GPU sequential training"""

    parser = argparse.ArgumentParser(
        description='AMSR2 GPU Sequential Super-Resolution Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:

1. Train on first 10 files:
   python gpu_sequential_amsr2.py --npz-dir /path/to/data --max-files 10

2. Train with specific batch size:
   python gpu_sequential_amsr2.py --npz-dir /path/to/data --batch-size 8

3. Quick test on 5 files:
   python gpu_sequential_amsr2.py --npz-dir /path/to/data --max-files 5 --epochs-per-file 1
        '''
    )

    # Required parameters
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='Path to directory with NPZ files')

    # Data parameters
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to train on (default: all)')
    parser.add_argument('--orbit-filter', choices=['A', 'D', 'U'],
                        help='Filter by orbit type (A=Ascending, D=Descending, U=Unknown)')

    # Training parameters
    parser.add_argument('--epochs-per-file', type=int, default=2,
                        help='Number of epochs per file (default: 2)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4 for 32GB GPU)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--scale-factor', type=int, default=10,
                        help='Super-resolution scale factor (default: 10)')

    # System parameters
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use automatic mixed precision (default: True for GPU)')
    parser.add_argument('--target-height', type=int, default=2000,
                        help='Target height for AMSR2 (default: 2000)')
    parser.add_argument('--target-width', type=int, default=420,
                        help='Target width for AMSR2 (default: 420)')

    # Output parameters
    parser.add_argument('--save-path', type=str, default='best_amsr2_gpu_model.pth',
                        help='Path to save best model')

    args = parser.parse_args()

    print("🛰️ AMSR2 GPU SEQUENTIAL SUPER-RESOLUTION TRAINER")
    print("=" * 60)

    # Device check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # Clear cache at start
        torch.cuda.empty_cache()

        # Log current memory usage
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        logger.info(f"   Currently allocated: {allocated:.1f} GB")
    else:
        logger.warning("⚠️ No GPU available, using CPU")
        logger.warning("   Training will be much slower")
        args.use_amp = False  # Disable AMP for CPU

    # Memory check
    memory_info = psutil.virtual_memory()
    logger.info(f"💾 System RAM: {memory_info.available / 1024 ** 3:.1f} GB available "
                f"of {memory_info.total / 1024 ** 3:.1f} GB ({memory_info.percent:.1f}% used)")

    # Find NPZ files
    npz_files = find_npz_files(args.npz_dir, args.max_files)

    if not npz_files:
        logger.error("❌ No NPZ files found")
        sys.exit(1)

    # Create model
    logger.info("🧠 Creating model...")
    model = UNetResNetSuperResolution(
        in_channels=1,
        out_channels=1,
        scale_factor=args.scale_factor
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Parameters: {total_params:,}")
    logger.info(f"   Model size: {total_params * 4 / 1024 ** 2:.1f} MB")
    logger.info(f"   Scale factor: {args.scale_factor}x")

    # Create preprocessor
    preprocessor = AMSR2NPZDataPreprocessor(
        target_height=args.target_height,
        target_width=args.target_width
    )

    # Create trainer
    trainer = GPUSequentialAMSR2Trainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        use_amp=args.use_amp
    )

    # Configuration summary
    logger.info(f"⚙️ Training configuration:")
    logger.info(f"   Files to train: {len(npz_files)}")
    logger.info(f"   Epochs per file: {args.epochs_per_file}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Learning rate: {args.lr}")
    logger.info(f"   Target size: {args.target_height}x{args.target_width}")
    logger.info(f"   Mixed precision: {args.use_amp}")

    if args.orbit_filter:
        logger.info(f"   Orbit filter: {args.orbit_filter}")

    # Estimate time
    estimated_batches = len(npz_files) * args.epochs_per_file * 10  # Rough estimate
    estimated_seconds = estimated_batches * (0.5 if device.type == 'cuda' else 2.0)
    logger.info(f"⏱️ Estimated training time: {estimated_seconds / 3600:.1f} hours")

    # User confirmation
    if len(npz_files) > 50:
        try:
            confirm = input("\n🚀 Start training? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                logger.info("❌ Training cancelled by user")
                sys.exit(0)
        except KeyboardInterrupt:
            logger.info("\n❌ Training cancelled by user")
            sys.exit(0)

    # Start training
    logger.info("\n🚀 Starting GPU sequential training...")
    start_time = time.time()

    try:
        training_history = trainer.train_sequential(
            npz_files=npz_files,
            preprocessor=preprocessor,
            epochs_per_file=args.epochs_per_file,
            batch_size=args.batch_size,
            save_path=args.save_path
        )

        end_time = time.time()
        training_time = end_time - start_time

        logger.info(f"\n🎉 Training completed!")
        logger.info(f"   Training time: {training_time / 3600:.2f} hours")
        logger.info(f"   Files processed: {len(training_history)}")

        # Create reports
        create_training_summary(training_history)
        plot_training_progress(training_history)

        logger.info(f"\n📁 Results:")
        logger.info(f"   Model: {args.save_path}")
        logger.info(f"   Summary: training_summary_gpu.json")
        logger.info(f"   Plot: training_progress_gpu.png")
        logger.info(f"   Log: amsr2_gpu_sequential.log")

    except KeyboardInterrupt:
        logger.info("\n⏹️ Training interrupted by user")
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("💾 GPU out of memory!")
            logger.error("   Try reducing batch_size or using gradient accumulation")
        else:
            logger.error(f"❌ Runtime error: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        logger.info("🛑 Program finished")


if __name__ == "__main__":
    main()