#!/usr/bin/env python3
"""
AMSR2 Brightness Temperature Super-Resolution - Complete Implementation
Полная реализация увеличения разрешения данных AMSR2 в 10 раз
Адаптировано под NPZ формат с множественными swath и scale factors
Оптимизировано для Xeon E5-2640 v4 (30 ядер максимум)

Автор: Volodymyr Didur
Версия: 2.1 - CPU Optimized Edition for Xeon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from tqdm import tqdm
import time
import json
import shutil
import argparse
from pathlib import Path
import multiprocessing
import psutil

# ====== CPU OPTIMIZATION FOR XEON E5-2640 v4 (30 CORES MAX) ======
# Настройка для 30 ядер (оставляем 10 для системы)
MAX_CPU_CORES = 30
os.environ['OMP_NUM_THREADS'] = str(MAX_CPU_CORES)
os.environ['MKL_NUM_THREADS'] = str(MAX_CPU_CORES)
os.environ['NUMEXPR_NUM_THREADS'] = str(MAX_CPU_CORES)

# Intel MKL оптимизации для Xeon
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'

# PyTorch настройки
torch.set_num_threads(MAX_CPU_CORES)
if hasattr(torch, 'set_num_interop_threads'):
    torch.set_num_interop_threads(15)  # Половина от MAX_CPU_CORES

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Лог информации об оптимизации
logger.info(f"🔧 CPU оптимизация для Xeon E5-2640 v4:")
logger.info(f"   Ограничение: {MAX_CPU_CORES} из {psutil.cpu_count()} доступных ядер")
logger.info(f"   Свободно для системы: {psutil.cpu_count() - MAX_CPU_CORES} ядер")
logger.info(f"   Доступная память: {psutil.virtual_memory().total / 1024 ** 3:.1f} GB")


# ================================================================================================
# SECTION 1: DATA PREPROCESSING AND LOADING
# ================================================================================================

class AMSR2NPZDataPreprocessor:
    """
    Препроцессор для AMSR2 NPZ данных в формате вашего процессора
    Обрабатывает множественные swath в одном файле и применяет scale factors
    """

    def __init__(self, target_height: int = None, target_width: int = None):
        self.target_height = target_height
        self.target_width = target_width
        self.stats = {
            'total_files': 0,
            'total_swaths': 0,
            'shapes_found': {},
            'orbit_types': {'A': 0, 'D': 0, 'U': 0},
            'scale_factors': [],
            'temp_ranges': []
        }

    def analyze_npz_files(self, npz_paths: List[str]) -> dict:
        """Анализ NPZ файлов для определения оптимальных размеров и характеристик"""

        logger.info(f"Анализируем {len(npz_paths)} NPZ файлов...")

        all_shapes = []

        for npz_path in tqdm(npz_paths[:10], desc="Анализ файлов"):  # Анализируем первые 10 файлов
            try:
                with np.load(npz_path, allow_pickle=True) as data:
                    swath_array = data['swath_array']

                    self.stats['total_files'] += 1

                    for swath_dict in swath_array:
                        swath = swath_dict.item() if isinstance(swath_dict, np.ndarray) else swath_dict

                        # Извлекаем данные
                        temperature = swath['temperature']
                        metadata = swath['metadata']

                        # Статистика
                        shape = temperature.shape
                        all_shapes.append(shape)

                        shape_key = f"{shape[0]}x{shape[1]}"
                        self.stats['shapes_found'][shape_key] = self.stats['shapes_found'].get(shape_key, 0) + 1

                        orbit_type = metadata.get('orbit_type', 'U')
                        self.stats['orbit_types'][orbit_type] += 1

                        scale_factor = metadata.get('scale_factor', 1.0)
                        self.stats['scale_factors'].append(scale_factor)

                        temp_range = metadata.get('temp_range', (0, 0))
                        self.stats['temp_ranges'].append(temp_range)

                        self.stats['total_swaths'] += 1

            except Exception as e:
                logger.error(f"Ошибка анализа {npz_path}: {e}")

        if all_shapes:
            heights = [s[0] for s in all_shapes]
            widths = [s[1] for s in all_shapes]

            analysis = {
                'total_files_analyzed': self.stats['total_files'],
                'total_swaths_found': self.stats['total_swaths'],
                'shapes_distribution': self.stats['shapes_found'],
                'orbit_distribution': self.stats['orbit_types'],
                'height_range': (min(heights), max(heights)),
                'width_range': (min(widths), max(widths)),
                'most_common_shape': max(self.stats['shapes_found'].items(), key=lambda x: x[1]),
                'recommended_size': (min(heights), min(widths)),
                'scale_factor_range': (min(self.stats['scale_factors']), max(self.stats['scale_factors'])),
                'temp_range_summary': {
                    'min_temp': min([tr[0] for tr in self.stats['temp_ranges']]),
                    'max_temp': max([tr[1] for tr in self.stats['temp_ranges']])
                }
            }

            # Устанавливаем рекомендуемые размеры если не заданы
            if self.target_height is None:
                self.target_height = analysis['recommended_size'][0]
            if self.target_width is None:
                self.target_width = analysis['recommended_size'][1]

            logger.info(f"Анализ завершен:")
            logger.info(f"  Найдено swath: {analysis['total_swaths_found']}")
            logger.info(f"  Размеры: {analysis['shapes_distribution']}")
            logger.info(f"  Orbit типы: {analysis['orbit_distribution']}")
            logger.info(f"  Рекомендуемый размер: {analysis['recommended_size']}")
            logger.info(f"  Установленный target: {self.target_height}x{self.target_width}")

            return analysis

        return {}

    def load_swath_from_npz(self, npz_path: str) -> List[Dict]:
        """Загрузка всех swath из одного NPZ файла"""

        try:
            with np.load(npz_path, allow_pickle=True) as data:
                swath_array = data['swath_array']
                period_info = data.get('period', 'Unknown period')

                swath_list = []

                for swath_dict in swath_array:
                    swath = swath_dict.item() if isinstance(swath_dict, np.ndarray) else swath_dict

                    # Извлекаем температуру и метаданные
                    raw_temperature = swath['temperature']
                    metadata = swath['metadata']

                    # Применяем scale factor
                    scale_factor = metadata.get('scale_factor', 1.0)

                    # Конвертируем в физические единицы
                    if raw_temperature.dtype != np.float32:
                        temperature = raw_temperature.astype(np.float32) * scale_factor
                    else:
                        temperature = raw_temperature * scale_factor

                    # Фильтруем невалидные значения (обычно 0 или очень маленькие)
                    temperature = np.where(temperature < 50, np.nan, temperature)  # Нереально низкие температуры
                    temperature = np.where(temperature > 350, np.nan, temperature)  # Нереально высокие температуры

                    swath_data = {
                        'temperature': temperature,
                        'metadata': metadata,
                        'file_info': {'source': npz_path, 'period': period_info}
                    }

                    swath_list.append(swath_data)

                return swath_list

        except Exception as e:
            logger.error(f"Ошибка загрузки {npz_path}: {e}")
            return []

    def crop_and_pad_to_target(self, temperature: np.ndarray) -> np.ndarray:
        """Обрезка или дополнение до target размера"""

        h, w = temperature.shape

        # Если размеры больше target - обрезаем по центру
        if h > self.target_height:
            start_h = (h - self.target_height) // 2
            temperature = temperature[start_h:start_h + self.target_height]

        if w > self.target_width:
            start_w = (w - self.target_width) // 2
            temperature = temperature[:, start_w:start_w + self.target_width]

        # Если размеры меньше target - дополняем с помощью reflection padding
        current_h, current_w = temperature.shape

        if current_h < self.target_height or current_w < self.target_width:
            pad_h = max(0, self.target_height - current_h)
            pad_w = max(0, self.target_width - current_w)

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            # Используем reflection padding для естественного продолжения
            temperature = np.pad(temperature,
                                 ((pad_top, pad_bottom), (pad_left, pad_right)),
                                 mode='reflect')

        return temperature

    def normalize_brightness_temperature(self, temperature: np.ndarray) -> np.ndarray:
        """Нормализация brightness temperature с учетом NaN значений"""

        # Заполняем NaN средним значением валидных пикселей
        valid_mask = ~np.isnan(temperature)
        if np.sum(valid_mask) > 0:
            mean_temp = np.mean(temperature[valid_mask])
            temperature = np.where(np.isnan(temperature), mean_temp, temperature)
        else:
            # Если все NaN, заполняем типичным значением
            temperature = np.full_like(temperature, 250.0)

        # Нормализация в диапазон [-1, 1]
        # Типичный диапазон AMSR2: 50-350K, центр около 200K
        temperature = np.clip(temperature, 50, 350)
        normalized = (temperature - 200) / 150

        return normalized.astype(np.float32)


class AMSR2NPZDataset(Dataset):
    """Dataset для AMSR2 NPZ данных с множественными swath"""

    def __init__(self, npz_paths: List[str], preprocessor: AMSR2NPZDataPreprocessor,
                 degradation_scale: int = 4, augment: bool = True,
                 filter_orbit_type: Optional[str] = None):

        self.npz_paths = npz_paths
        self.preprocessor = preprocessor
        self.degradation_scale = degradation_scale
        self.augment = augment
        self.filter_orbit_type = filter_orbit_type

        # Предварительно загружаем все swath для создания индекса
        self.swath_index = []

        logger.info("Индексирование swath данных...")

        for npz_path in tqdm(npz_paths, desc="Загрузка NPZ файлов"):
            swath_list = preprocessor.load_swath_from_npz(npz_path)

            for swath_idx, swath in enumerate(swath_list):
                # Фильтрация по типу орбиты если требуется
                if filter_orbit_type is not None:
                    orbit_type = swath['metadata'].get('orbit_type', 'U')
                    if orbit_type != filter_orbit_type:
                        continue

                # Проверяем валидность данных
                temp = swath['temperature']
                valid_pixels = np.sum(~np.isnan(temp))
                total_pixels = temp.size

                if valid_pixels / total_pixels > 0.1:  # Минимум 10% валидных пикселей
                    self.swath_index.append({
                        'npz_path': npz_path,
                        'swath_idx': swath_idx,
                        'swath_data': swath
                    })

        logger.info(f"Найдено {len(self.swath_index)} валидных swath")

        # Анализируем размеры если еще не было
        if not hasattr(preprocessor, 'target_height') or preprocessor.target_height is None:
            self.preprocessor.analyze_npz_files(npz_paths)

    def __len__(self):
        return len(self.swath_index)

    def create_degradation(self, high_res: np.ndarray) -> np.ndarray:
        """Создание деградированной версии для self-supervised обучения"""

        h, w = high_res.shape

        # 1. Downsample
        low_res = F.interpolate(
            torch.from_numpy(high_res).unsqueeze(0).unsqueeze(0),
            size=(h // self.degradation_scale, w // self.degradation_scale),
            mode='bilinear',
            align_corners=False
        )

        # 2. Добавляем реалистичный шум (характерный для AMSR2)
        noise_std = 0.01  # Меньше шума для научных данных
        noise = torch.randn_like(low_res) * noise_std
        low_res = low_res + noise

        # 3. Легкое размытие (PSF сенсора)
        blur = transforms.GaussianBlur(kernel_size=3, sigma=0.3)
        low_res = blur(low_res)

        # 4. Upsampling обратно к исходному размеру
        degraded = F.interpolate(
            low_res,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )

        return degraded.squeeze().numpy()

    def augment_data(self, data: np.ndarray) -> np.ndarray:
        """Аугментация с сохранением геофизических свойств"""

        if not self.augment:
            return data

        # Только safe аугментации для научных данных

        # Случайное отражение (географически корректно)
        if np.random.rand() > 0.5:
            data = np.fliplr(data)

        if np.random.rand() > 0.5:
            data = np.flipud(data)

        # Поворот только на 90 градусов (сохраняет grid структуру)
        if np.random.rand() > 0.7:
            k = np.random.choice([1, 2, 3])
            data = np.rot90(data, k)

        return data

    def __getitem__(self, idx):

        swath_info = self.swath_index[idx]
        swath_data = swath_info['swath_data']

        # Извлекаем температуру
        temperature = swath_data['temperature'].copy()

        # Обрезка/дополнение до target размера
        temperature = self.preprocessor.crop_and_pad_to_target(temperature)

        # Нормализация
        temperature = self.preprocessor.normalize_brightness_temperature(temperature)

        # Аугментация
        temperature = self.augment_data(temperature)

        # Создание деградированной версии
        degraded = self.create_degradation(temperature)

        # Конвертация в тензоры
        high_res = torch.from_numpy(temperature).unsqueeze(0).float()
        low_res = torch.from_numpy(degraded).unsqueeze(0).float()

        return low_res, high_res


# ================================================================================================
# SECTION 2: NEURAL NETWORK ARCHITECTURE
# ================================================================================================

class ResNetBlock(nn.Module):
    """ResNet блок для encoder"""

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
    """ResNet-based encoder для U-Net"""

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
    """Decoder для U-Net с upsampling"""

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
    """U-Net с ResNet backbone для AMSR2 super-resolution"""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, scale_factor: int = 10):
        super().__init__()

        self.scale_factor = scale_factor
        self.encoder = UNetResNetEncoder(in_channels)
        self.decoder = UNetDecoder(out_channels)

        # Progressive upsampling для больших scale factors
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


# ================================================================================================
# SECTION 3: LOSS FUNCTIONS AND METRICS
# ================================================================================================

class AMSR2SpecificLoss(nn.Module):
    """Loss функция специально для AMSR2 brightness temperature"""

    def __init__(self, alpha: float = 1.0, beta: float = 0.15, gamma: float = 0.05):
        super().__init__()
        self.alpha = alpha  # L1 loss
        self.beta = beta  # Gradient loss
        self.gamma = gamma  # Physical consistency

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Сохранение градиентов важно для temperature структур"""

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
        """Проверка физической корректности brightness temperature"""

        # 1. Сохранение средней температуры
        pred_mean = torch.mean(pred, dim=[2, 3])
        target_mean = torch.mean(target, dim=[2, 3])
        energy_loss = self.mse_loss(pred_mean, target_mean)

        # 2. Сохранение распределения температур
        pred_std = torch.std(pred, dim=[2, 3])
        target_std = torch.std(target, dim=[2, 3])
        distribution_loss = self.mse_loss(pred_std, target_std)

        # 3. Penalize unrealistic values (outside normalized range)
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


class SuperResolutionMetrics:
    """Класс для вычисления метрик качества super-resolution"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.psnr_values = []
        self.ssim_values = []
        self.mse_values = []
        self.mae_values = []

    def psnr(self, pred: np.ndarray, target: np.ndarray, max_val: float = 2.0) -> float:
        """Peak Signal-to-Noise Ratio"""
        mse = np.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_val / np.sqrt(mse))

    def ssim(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Simplified SSIM calculation"""
        mu1 = np.mean(pred)
        mu2 = np.mean(target)

        sigma1_sq = np.var(pred)
        sigma2_sq = np.var(target)
        sigma12 = np.mean((pred - mu1) * (target - mu2))

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)

        return numerator / denominator

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Обновление метрик"""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        batch_size = pred_np.shape[0]

        for i in range(batch_size):
            p = pred_np[i, 0]  # Убираем channel dimension
            t = target_np[i, 0]

            self.psnr_values.append(self.psnr(p, t))
            self.ssim_values.append(self.ssim(p, t))
            self.mse_values.append(mean_squared_error(t.flatten(), p.flatten()))
            self.mae_values.append(mean_absolute_error(t.flatten(), p.flatten()))

    def get_metrics(self) -> dict:
        """Получение средних метрик"""
        if not self.psnr_values:
            return {}

        return {
            'PSNR': np.mean(self.psnr_values),
            'SSIM': np.mean(self.ssim_values),
            'MSE': np.mean(self.mse_values),
            'MAE': np.mean(self.mae_values),
            'PSNR_std': np.std(self.psnr_values),
            'SSIM_std': np.std(self.ssim_values)
        }


# ================================================================================================
# SECTION 4: TRAINING INFRASTRUCTURE
# ================================================================================================

class AMSR2SuperResolutionTrainer:
    """Тренер для обучения модели super-resolution"""

    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device

        # Оптимизатор и scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Loss функция и метрики
        self.criterion = AMSR2SpecificLoss()
        self.metrics = SuperResolutionMetrics()

        # Статистики обучения
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

        # Лучшая модель
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Обучение на одной эпохе с мониторингом CPU"""
        self.model.train()
        epoch_losses = []
        self.metrics.reset()

        # CPU мониторинг для отслеживания эффективности
        cpu_monitoring = self.device.type == 'cpu'
        if cpu_monitoring:
            try:
                import psutil
                cpu_percent_start = psutil.cpu_percent()
                memory_start = psutil.virtual_memory().percent
                process = psutil.Process()
                start_threads = process.num_threads()
                logger.info(
                    f"🔧 Начало эпохи - CPU: {cpu_percent_start:.1f}%, RAM: {memory_start:.1f}%, Threads: {start_threads}")
            except ImportError:
                cpu_monitoring = False
                logger.warning("psutil не установлен - мониторинг CPU недоступен")

        progress_bar = tqdm(train_loader, desc="Обучение")

        for batch_idx, (low_res, high_res) in enumerate(progress_bar):
            low_res = low_res.to(self.device)
            high_res = high_res.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(low_res)

            # Вычисление loss
            loss, loss_components = self.criterion(pred, high_res)

            # Backward pass
            loss.backward()

            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Статистика
            epoch_losses.append(loss.item())
            self.metrics.update(pred, high_res)

            # Обновление progress bar с расширенной информацией
            if batch_idx % 10 == 0:
                current_metrics = self.metrics.get_metrics()
                postfix = {
                    'Loss': f"{loss.item():.4f}",
                    'PSNR': f"{current_metrics.get('PSNR', 0):.2f}",
                    'SSIM': f"{current_metrics.get('SSIM', 0):.3f}"
                }

                # Добавляем CPU статистику каждые 50 батчей для мониторинга
                if cpu_monitoring and batch_idx % 50 == 0:
                    try:
                        cpu_usage = psutil.cpu_percent(interval=0.1)
                        memory_usage = psutil.virtual_memory().percent
                        process_threads = process.num_threads()

                        # Добавляем в progress bar только ключевые метрики
                        postfix['CPU'] = f"{cpu_usage:.0f}%"
                        postfix['RAM'] = f"{memory_usage:.0f}%"

                        # Детальный лог каждые 100 батчей
                        if batch_idx % 100 == 0:
                            logger.info(
                                f"Batch {batch_idx}: CPU {cpu_usage:.1f}%, RAM {memory_usage:.1f}%, Threads {process_threads}")
                    except:
                        pass  # Игнорируем ошибки мониторинга

                progress_bar.set_postfix(postfix)

        # Финальная статистика эпохи
        if cpu_monitoring:
            try:
                cpu_percent_end = psutil.cpu_percent()
                memory_end = psutil.virtual_memory().percent
                end_threads = process.num_threads()
                logger.info(
                    f"🏁 Конец эпохи - CPU: {cpu_percent_end:.1f}%, RAM: {memory_end:.1f}%, Threads: {end_threads}")
            except:
                pass

        # Средние метрики за эпоху
        epoch_metrics = self.metrics.get_metrics()
        avg_loss = np.mean(epoch_losses)

        return {
            'avg_loss': avg_loss,
            'metrics': epoch_metrics,
            'loss_components': loss_components
        }

    def validate_epoch(self, val_loader: DataLoader) -> dict:
        """Валидация на одной эпохе"""
        self.model.eval()
        epoch_losses = []
        self.metrics.reset()

        with torch.no_grad():
            for low_res, high_res in tqdm(val_loader, desc="Валидация"):
                low_res = low_res.to(self.device)
                high_res = high_res.to(self.device)

                # Forward pass
                pred = self.model(low_res)
                loss, loss_components = self.criterion(pred, high_res)

                epoch_losses.append(loss.item())
                self.metrics.update(pred, high_res)

        # Средние метрики за эпоху
        epoch_metrics = self.metrics.get_metrics()
        avg_loss = np.mean(epoch_losses)

        return {
            'avg_loss': avg_loss,
            'metrics': epoch_metrics,
            'loss_components': loss_components
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, save_path: str = "best_amsr2_model.pth"):
        """Полный цикл обучения"""

        logger.info(f"Начинаем обучение на {num_epochs} эпох")
        logger.info(f"Размер модели: {sum(p.numel() for p in self.model.parameters()):,} параметров")

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Обучение
            train_results = self.train_epoch(train_loader)

            # Валидация
            val_results = self.validate_epoch(val_loader)

            # Сохранение статистик
            self.train_losses.append(train_results['avg_loss'])
            self.val_losses.append(val_results['avg_loss'])
            self.train_metrics.append(train_results['metrics'])
            self.val_metrics.append(val_results['metrics'])

            # Scheduler step
            self.scheduler.step(val_results['avg_loss'])

            # Сохранение лучшей модели
            if val_results['avg_loss'] < self.best_val_loss:
                self.best_val_loss = val_results['avg_loss']
                self.best_model_state = self.model.state_dict().copy()
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': self.best_val_loss,
                    'train_metrics': self.train_metrics,
                    'val_metrics': self.val_metrics
                }, save_path)
                logger.info(f"Сохранена лучшая модель с val_loss: {self.best_val_loss:.4f}")

            epoch_time = time.time() - epoch_start

            # Логирование результатов эпохи
            logger.info(f"\nЭпоха {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
            logger.info(f"Train Loss: {train_results['avg_loss']:.4f}")
            logger.info(f"Val Loss: {val_results['avg_loss']:.4f}")

            if train_results['metrics']:
                logger.info(
                    f"Train PSNR: {train_results['metrics']['PSNR']:.2f} ± {train_results['metrics']['PSNR_std']:.2f}")
                logger.info(
                    f"Train SSIM: {train_results['metrics']['SSIM']:.3f} ± {train_results['metrics']['SSIM_std']:.3f}")

            if val_results['metrics']:
                logger.info(
                    f"Val PSNR: {val_results['metrics']['PSNR']:.2f} ± {val_results['metrics']['PSNR_std']:.2f}")
                logger.info(
                    f"Val SSIM: {val_results['metrics']['SSIM']:.3f} ± {val_results['metrics']['SSIM_std']:.3f}")

        total_time = time.time() - start_time
        logger.info(f"\nОбучение завершено за {total_time / 3600:.1f} часов")
        logger.info(f"Лучший val_loss: {self.best_val_loss:.4f}")

        # Загружаем лучшую модель
        self.model.load_state_dict(self.best_model_state)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss
        }

    def plot_training_history(self, save_path: str = "training_history.png"):
        """Визуализация процесса обучения"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss график
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # PSNR график
        if self.train_metrics and self.val_metrics:
            train_psnr = [m.get('PSNR', 0) for m in self.train_metrics]
            val_psnr = [m.get('PSNR', 0) for m in self.val_metrics]

            axes[0, 1].plot(epochs, train_psnr, 'b-', label='Train PSNR')
            axes[0, 1].plot(epochs, val_psnr, 'r-', label='Val PSNR')
            axes[0, 1].set_title('PSNR Over Time')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('PSNR (dB)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # SSIM график
            train_ssim = [m.get('SSIM', 0) for m in self.train_metrics]
            val_ssim = [m.get('SSIM', 0) for m in self.val_metrics]

            axes[1, 0].plot(epochs, train_ssim, 'b-', label='Train SSIM')
            axes[1, 0].plot(epochs, val_ssim, 'r-', label='Val SSIM')
            axes[1, 0].set_title('SSIM Over Time')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('SSIM')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            # MAE график
            train_mae = [m.get('MAE', 0) for m in self.train_metrics]
            val_mae = [m.get('MAE', 0) for m in self.val_metrics]

            axes[1, 1].plot(epochs, train_mae, 'b-', label='Train MAE')
            axes[1, 1].plot(epochs, val_mae, 'r-', label='Val MAE')
            axes[1, 1].set_title('MAE Over Time')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"График сохранен как {save_path}")


# ================================================================================================
# SECTION 5: DATA LOADING AND UTILITIES
# ================================================================================================

def create_amsr2_data_loaders(npz_dir: str, batch_size: int = 6,
                              val_split: float = 0.2, num_workers: int = 15,  # Оптимизировано для 30 ядер
                              filter_orbit_type: Optional[str] = None) -> tuple:
    """Создание data loaders для NPZ файлов AMSR2 с CPU оптимизацией"""

    # Поиск всех NPZ файлов
    npz_pattern = os.path.join(npz_dir, "*.npz")
    npz_files = glob.glob(npz_pattern)

    if not npz_files:
        raise ValueError(f"Не найдено NPZ файлов в {npz_dir}")

    logger.info(f"Найдено {len(npz_files)} NPZ файлов")

    # Разделение на train/val
    split_idx = int(len(npz_files) * (1 - val_split))
    train_files = npz_files[:split_idx]
    val_files = npz_files[split_idx:]

    logger.info(f"Train files: {len(train_files)}")
    logger.info(f"Val files: {len(val_files)}")

    # Создание preprocessor с анализом данных
    preprocessor = AMSR2NPZDataPreprocessor()
    analysis = preprocessor.analyze_npz_files(npz_files[:5])  # Анализируем первые 5 файлов

    # Создание datasets
    train_dataset = AMSR2NPZDataset(
        npz_paths=train_files,
        preprocessor=preprocessor,
        degradation_scale=4,
        augment=True,
        filter_orbit_type=filter_orbit_type
    )

    val_dataset = AMSR2NPZDataset(
        npz_paths=val_files,
        preprocessor=preprocessor,
        degradation_scale=4,
        augment=False,
        filter_orbit_type=filter_orbit_type
    )

    # Ограничение workers для стабильности системы
    effective_workers = min(num_workers, 15)  # Максимум 15 workers

    logger.info(f"🔧 DataLoader настройки:")
    logger.info(f"   Workers: {effective_workers} (ограничено для стабильности)")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Pin memory: False (CPU mode)")

    # Создание data loaders с CPU оптимизацией
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=False,  # Отключено для CPU
        drop_last=True,
        persistent_workers=True  # Эффективность для длительного обучения
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=False,  # Отключено для CPU
        persistent_workers=True
    )

    return train_loader, val_loader, preprocessor, analysis


def create_test_npz_files(output_dir: str, num_files: int = 5):
    """Создание тестовых NPZ файлов в формате AMSR2 процессора"""

    logger.info(f"Создание {num_files} тестовых NPZ файлов...")

    for file_idx in range(num_files):
        swath_array = []

        # Каждый файл содержит несколько swath
        num_swaths = np.random.randint(5, 15)

        for swath_idx in range(num_swaths):
            # Создаем реалистичные размеры
            height = np.random.choice([2041, 2036, 2000, 1950])
            width = np.random.choice([421, 420, 422])

            # Создаем raw temperature данные (до применения scale factor)
            # Типичные raw значения до scaling
            raw_temp = np.random.randint(500, 3000, size=(height, width), dtype=np.uint16)

            # Добавляем структуры
            x, y = np.meshgrid(np.linspace(0, 10, width), np.linspace(0, 20, height))
            structures = (200 * np.sin(x / 2) * np.cos(y / 3)).astype(np.int16)
            raw_temp = (raw_temp.astype(np.int32) + structures).astype(np.uint16)

            # Метаданные как в вашем процессоре
            scale_factor = 0.1  # Типичный scale factor для AMSR2
            orbit_type = np.random.choice(['A', 'D', 'U'])

            # После применения scale factor получаем реалистичные BT
            actual_temp = raw_temp.astype(np.float32) * scale_factor
            temp_range = (int(np.min(actual_temp)), int(np.max(actual_temp)))

            metadata = {
                'orbit_type': orbit_type,
                'scale_factor': scale_factor,
                'temp_range': temp_range,
                'shape': raw_temp.shape
            }

            swath_dict = {
                'temperature': raw_temp,  # Raw data как в вашем формате
                'metadata': metadata
            }

            swath_array.append(swath_dict)

        # Сохраняем в NPZ формате
        period_info = f"2024-01-{file_idx + 1:02d}T00:00:00 to 2024-01-{file_idx + 1:02d}T23:59:59"

        save_dict = {
            'swath_array': np.array(swath_array, dtype=object),
            'period': period_info
        }

        output_file = os.path.join(output_dir, f"AMSR2_temp_only_test_{file_idx:03d}.npz")
        np.savez_compressed(output_file, **save_dict)

        logger.info(f"Создан тестовый файл: {output_file} ({num_swaths} swaths)")


def test_model_inference(model: nn.Module, test_loader: DataLoader,
                         device: torch.device, save_examples: bool = True,
                         preprocessor: AMSR2NPZDataPreprocessor = None):
    """Тестирование модели с визуализацией results для AMSR2"""

    model.eval()
    metrics = SuperResolutionMetrics()

    example_count = 0
    max_examples = 5

    with torch.no_grad():
        for batch_idx, (low_res, high_res) in enumerate(test_loader):
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            # Inference
            start_time = time.time()
            pred = model(low_res)
            inference_time = time.time() - start_time

            # Метрики
            metrics.update(pred, high_res)

            # Сохранение примеров с денормализацией
            if save_examples and example_count < max_examples:
                batch_size = low_res.shape[0]
                for i in range(min(batch_size, max_examples - example_count)):

                    lr_img = low_res[i, 0].cpu().numpy()
                    hr_img = high_res[i, 0].cpu().numpy()
                    pred_img = pred[i, 0].cpu().numpy()

                    # Денормализация для реалистичных значений BT
                    lr_bt = (lr_img * 150) + 200  # Обратная нормализация
                    hr_bt = (hr_img * 150) + 200
                    pred_bt = (pred_img * 150) + 200

                    # Создание сравнительного изображения
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                    # Input (degraded)
                    im1 = axes[0].imshow(lr_bt, cmap='coolwarm', vmin=150, vmax=300, aspect='auto')
                    axes[0].set_title(f'Input (Degraded)\n{lr_bt.shape}\nRange: {lr_bt.min():.1f}-{lr_bt.max():.1f}K')
                    axes[0].axis('off')
                    plt.colorbar(im1, ax=axes[0], label='Brightness Temperature (K)')

                    # Target (original)
                    im2 = axes[1].imshow(hr_bt, cmap='coolwarm', vmin=150, vmax=300, aspect='auto')
                    axes[1].set_title(f'Target (Original)\n{hr_bt.shape}\nRange: {hr_bt.min():.1f}-{hr_bt.max():.1f}K')
                    axes[1].axis('off')
                    plt.colorbar(im2, ax=axes[1], label='Brightness Temperature (K)')

                    # Enhanced
                    im3 = axes[2].imshow(pred_bt, cmap='coolwarm', vmin=150, vmax=300, aspect='auto')
                    axes[2].set_title(
                        f'Enhanced (10x SR)\n{pred_bt.shape}\nRange: {pred_bt.min():.1f}-{pred_bt.max():.1f}K')
                    axes[2].axis('off')
                    plt.colorbar(im3, ax=axes[2], label='Brightness Temperature (K)')

                    # Метрики для этого примера
                    psnr_val = metrics.psnr(pred_img, hr_img)
                    ssim_val = metrics.ssim(pred_img, hr_img)

                    plt.suptitle(
                        f'AMSR2 Super-Resolution Example {example_count + 1}\nPSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.3f}',
                        fontsize=14)
                    plt.tight_layout()
                    plt.savefig(f'amsr2_example_{example_count + 1}.png', dpi=300, bbox_inches='tight')
                    plt.show()

                    example_count += 1

                    if example_count >= max_examples:
                        break

            if batch_idx == 0:
                logger.info(f"Время inference для batch {low_res.shape}: {inference_time:.3f}s")

            if example_count >= max_examples:
                break

    # Финальные метрики
    final_metrics = metrics.get_metrics()

    logger.info(f"\n=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ AMSR2 ===")
    logger.info(f"PSNR: {final_metrics['PSNR']:.2f} ± {final_metrics['PSNR_std']:.2f} dB")
    logger.info(f"SSIM: {final_metrics['SSIM']:.3f} ± {final_metrics['SSIM_std']:.3f}")
    logger.info(f"MSE: {final_metrics['MSE']:.6f}")
    logger.info(f"MAE: {final_metrics['MAE']:.6f}")

    return final_metrics


def process_new_npz_file(model_path: str, npz_path: str, output_path: str,
                         swath_index: int = None, orbit_filter: str = None):
    """Обработка нового NPZ файла с trained моделью"""

    logger.info(f"Обработка NPZ файла: {npz_path}")

    # Загрузка модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetResNetSuperResolution()

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Загрузка данных
    preprocessor = AMSR2NPZDataPreprocessor()
    swath_list = preprocessor.load_swath_from_npz(npz_path)

    logger.info(f"Загружено {len(swath_list)} swath из файла")

    enhanced_swaths = []

    for idx, swath in enumerate(swath_list):
        # Фильтрация по индексу если указан
        if swath_index is not None and idx != swath_index:
            continue

        # Фильтрация по типу орбиты если указан
        if orbit_filter is not None:
            orbit_type = swath['metadata'].get('orbit_type', 'U')
            if orbit_type != orbit_filter:
                continue

        logger.info(f"Обработка swath {idx}: {swath['metadata']}")

        # Подготовка данных
        temperature = swath['temperature'].copy()
        original_shape = temperature.shape

        # Применяем scale factor если есть raw данные
        scale_factor = swath['metadata'].get('scale_factor', 1.0)
        if temperature.dtype != np.float32:
            temperature = temperature.astype(np.float32) * scale_factor

        # Фильтрация невалидных значений
        temperature = np.where(temperature < 50, np.nan, temperature)
        temperature = np.where(temperature > 350, np.nan, temperature)

        # Препроцессинг
        if preprocessor.target_height is None:
            preprocessor.target_height, preprocessor.target_width = temperature.shape

        processed_temp = preprocessor.crop_and_pad_to_target(temperature)
        normalized_temp = preprocessor.normalize_brightness_temperature(processed_temp)

        # Inference
        input_tensor = torch.from_numpy(normalized_temp).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            enhanced = model(input_tensor)

        # Денормализация
        enhanced_np = enhanced.squeeze().cpu().numpy()
        enhanced_bt = (enhanced_np * 150) + 200  # Обратная нормализация

        enhanced_swaths.append({
            'original_shape': original_shape,
            'enhanced_shape': enhanced_bt.shape,
            'enhanced_temperature': enhanced_bt,
            'metadata': swath['metadata'],
            'enhancement_info': {
                'model_used': model_path,
                'scale_factor_applied': 10,
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        })

        logger.info(f"  Enhanced: {original_shape} → {enhanced_bt.shape}")

    # Сохранение результатов
    if enhanced_swaths:
        np.savez_compressed(output_path, enhanced_swaths=enhanced_swaths)
        logger.info(f"Результаты сохранены: {output_path}")
        logger.info(f"Обработано {len(enhanced_swaths)} swath")
    else:
        logger.warning("Нет данных для сохранения после фильтрации")

    return enhanced_swaths


# ================================================================================================
# SECTION 6: MAIN EXECUTION
# ================================================================================================

def main():
    """Основная функция для обучения с NPZ данными - CPU оптимизированная версия"""

    parser = argparse.ArgumentParser(description='AMSR2 Super-Resolution Training - Xeon Optimized')

    # Основные параметры
    parser.add_argument('--npz-dir', type=str, default='./npz_data',
                        help='Путь к папке с NPZ файлами')
    parser.add_argument('--batch-size', type=int, default=6,
                        help='Размер batch (оптимизировано для 30 ядер)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Количество эпох')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--scale-factor', type=int, default=10,
                        help='Масштаб увеличения')
    parser.add_argument('--orbit-filter', choices=['A', 'D', 'U'],
                        help='Фильтр по типу орбиты')
    parser.add_argument('--test-mode', action='store_true',
                        help='Режим тестирования с синтетическими данными')

    # CPU оптимизация параметры
    parser.add_argument('--cpu-workers', type=int, default=15,
                        help='Количество CPU workers для DataLoader (макс 15)')
    parser.add_argument('--cpu-threads', type=int, default=30,
                        help='Количество CPU threads для PyTorch (макс 30)')
    parser.add_argument('--max-cpu-cores', type=int, default=30,
                        help='Максимальное количество CPU ядер для использования')

    # Дополнительные параметры
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Доля данных для валидации')
    parser.add_argument('--save-path', type=str, default='best_amsr2_model.pth',
                        help='Путь для сохранения лучшей модели')

    args = parser.parse_args()

    # Применение ограничений CPU
    max_cores = min(args.max_cpu_cores, 30)  # Жесткое ограничение на 30 ядер
    max_workers = min(args.cpu_workers, 15)  # Жесткое ограничение на 15 workers

    if args.cpu_threads:
        actual_threads = min(args.cpu_threads, max_cores)
        torch.set_num_threads(actual_threads)
        os.environ['OMP_NUM_THREADS'] = str(actual_threads)
        os.environ['MKL_NUM_THREADS'] = str(actual_threads)
        logger.info(f"✅ Установлено CPU threads: {actual_threads}")

    # Настройки
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    SCALE_FACTOR = args.scale_factor

    # Информация о системе и оптимизации
    logger.info(f"🖥️ Используется устройство: {DEVICE}")
    logger.info(f"⚙️  Параметры обучения: batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}, lr={LEARNING_RATE}")

    if DEVICE.type == 'cpu':
        logger.info("🔧 CPU оптимизации для Xeon E5-2640 v4:")
        logger.info(f"   Используется ядер: {max_cores} из {psutil.cpu_count()} доступных")
        logger.info(f"   DataLoader workers: {max_workers}")
        logger.info(f"   Системе оставлено: {psutil.cpu_count() - max_cores} ядер")
        logger.info(f"   Доступная память: {psutil.virtual_memory().total / 1024 ** 3:.1f} GB")

        # Проверка оптимальности настроек
        if BATCH_SIZE > 8:
            logger.warning(f"⚠️  Batch size {BATCH_SIZE} может быть слишком большим для CPU")
        if max_workers > 20:
            logger.warning(f"⚠️  Workers {max_workers} могут перегрузить систему")

    # Путь к NPZ данным
    NPZ_DIR = args.npz_dir

    # Проверяем существование директории или создаем тестовые данные
    if not os.path.exists(NPZ_DIR) or args.test_mode:
        if args.test_mode:
            logger.info("🧪 Режим тестирования: создаем синтетические NPZ файлы...")
        else:
            logger.info("📁 Директория не найдена, создаем тестовые NPZ файлы для демонстрации...")

        test_dir = "test_npz_data"
        os.makedirs(test_dir, exist_ok=True)

        # Создаем меньше файлов для быстрого тестирования на CPU
        num_test_files = 3 if args.test_mode else 5
        create_test_npz_files(test_dir, num_files=num_test_files)
        NPZ_DIR = test_dir
        logger.info(f"✅ Тестовые данные созданы в {NPZ_DIR}")

    # Создание data loaders с CPU оптимизацией
    try:
        train_loader, val_loader, preprocessor, analysis = create_amsr2_data_loaders(
            npz_dir=NPZ_DIR,
            batch_size=BATCH_SIZE,
            val_split=args.val_split,
            num_workers=max_workers,  # Используем ограниченное количество workers
            filter_orbit_type=args.orbit_filter
        )

        logger.info(f"📊 Статистика данных:")
        logger.info(f"   Total swaths: {analysis.get('total_swaths_found', 0)}")
        logger.info(f"   Train swaths: {len(train_loader.dataset)}")
        logger.info(f"   Val swaths: {len(val_loader.dataset)}")
        logger.info(f"   Orbit distribution: {analysis.get('orbit_distribution', {})}")
        logger.info(f"   Target size: {preprocessor.target_height}x{preprocessor.target_width}")

        # Оценка времени обучения для CPU
        if DEVICE.type == 'cpu':
            total_swaths = len(train_loader.dataset)
            estimated_time_hours = (total_swaths * NUM_EPOCHS) / (max_cores * 50)  # Примерная оценка
            logger.info(f"⏱️  Примерное время обучения: {estimated_time_hours:.1f} часов")
            if estimated_time_hours > 100:
                logger.warning(
                    "⚠️  Обучение займет очень много времени. Рассмотрите уменьшение epochs или использование GPU")

    except Exception as e:
        logger.error(f"❌ Ошибка создания data loaders: {e}")
        return

    # Создание модели
    model = UNetResNetSuperResolution(
        in_channels=1,
        out_channels=1,
        scale_factor=SCALE_FACTOR
    )

    # Информация о модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024  # float32

    logger.info(f"🧠 Информация о модели:")
    logger.info(f"   Общие параметры: {total_params:,}")
    logger.info(f"   Обучаемые параметры: {trainable_params:,}")
    logger.info(f"   Размер модели: {model_size_mb:.1f} MB")
    logger.info(f"   Scale factor: {SCALE_FACTOR}x")

    # Создание тренера
    trainer = AMSR2SuperResolutionTrainer(
        model=model,
        device=DEVICE,
        learning_rate=LEARNING_RATE
    )

    # Информация о старте обучения
    logger.info(f"🚀 Начинаем обучение:")
    logger.info(f"   Эпохи: {NUM_EPOCHS}")
    logger.info(f"   Batch size: {BATCH_SIZE}")
    logger.info(f"   Learning rate: {LEARNING_RATE}")
    logger.info(f"   Сохранение модели: {args.save_path}")

    # Обучение
    training_start_time = time.time()
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        save_path=args.save_path
    )
    training_end_time = time.time()

    total_training_time = training_end_time - training_start_time
    logger.info(f"⏱️  Общее время обучения: {total_training_time / 3600:.2f} часов")

    # Визуализация результатов
    trainer.plot_training_history("amsr2_training_history.png")

    # Тестирование модели
    test_metrics = test_model_inference(
        model=model,
        test_loader=val_loader,
        device=DEVICE,
        save_examples=True,
        preprocessor=preprocessor
    )
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        save_path="best_amsr2_model.pth"
    )

    # Визуализация результатов
    trainer.plot_training_history("amsr2_training_history.png")

    # Сохранение финальной статистики с системной информацией
    final_stats = {
        'training_results': training_results,
        'test_metrics': test_metrics,
        'data_analysis': analysis,
        'model_info': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'scale_factor': SCALE_FACTOR,
            'target_size': f"{preprocessor.target_height}x{preprocessor.target_width}"
        },
        'training_config': {
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'orbit_filter': args.orbit_filter,
            'device': str(DEVICE),
            'total_training_time_hours': total_training_time / 3600
        },
        'system_info': {
            'cpu_cores_total': psutil.cpu_count(),
            'cpu_cores_used': max_cores,
            'cpu_cores_free': psutil.cpu_count() - max_cores,
            'dataloader_workers': max_workers,
            'memory_total_gb': psutil.virtual_memory().total / 1024 ** 3,
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            'pytorch_version': torch.__version__
        }
    }

    with open('amsr2_training_summary.json', 'w') as f:
        json.dump(final_stats, f, indent=2, default=str)

    # Финальный отчет
    logger.info("🎉 Обучение завершено успешно!")
    logger.info("📁 Файлы результатов:")
    logger.info(f"   - {args.save_path} (веса модели)")
    logger.info("   - amsr2_training_history.png (графики)")
    logger.info("   - amsr2_training_summary.json (статистика)")
    logger.info("   - amsr2_example_*.png (примеры результатов)")

    # Рекомендации на основе результатов
    if DEVICE.type == 'cpu':
        final_psnr = test_metrics.get('PSNR', 0)
        if final_psnr > 30:
            logger.info("✅ Отличные результаты! PSNR > 30 dB")
        elif final_psnr > 25:
            logger.info("✅ Хорошие результаты! PSNR > 25 dB")
        else:
            logger.info("⚠️ Результаты можно улучшить. Попробуйте:")
            logger.info("   - Увеличить количество эпох")
            logger.info("   - Уменьшить learning rate")
            logger.info("   - Добавить больше данных")

    # Информация о следующих шагах
    logger.info("\n📝 Следующие шаги:")
    logger.info("   1. Проанализируйте графики в amsr2_training_history.png")
    logger.info("   2. Проверьте примеры результатов (amsr2_example_*.png)")
    logger.info("   3. Используйте модель для inference на новых данных:")
    logger.info(f"      process_new_npz_file('{args.save_path}', 'new_data.npz', 'result.npz')")


if __name__ == "__main__":
    main()