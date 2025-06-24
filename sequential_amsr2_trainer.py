#!/usr/bin/env python3
"""
AMSR2 Sequential Trainer - Memory Protected Edition
Последовательная обработка файлов с защитой от переполнения памяти
Обучение происходит файл за файлом, без предварительной загрузки всех данных

Автор: Volodymyr Didur
Версия: 4.0 - Sequential Processing Edition
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
import time
import json
import argparse
from pathlib import Path
import psutil
import threading
import signal
import sys
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# ====== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ======
EMERGENCY_STOP = False
MEMORY_THRESHOLD = 80.0
MONITOR_INTERVAL = 6

# ====== НАСТРОЙКА ЛОГИРОВАНИЯ ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('amsr2_sequential.log')
    ]
)
logger = logging.getLogger(__name__)


# ====== МОНИТОРИНГ ПАМЯТИ ======
class MemoryMonitor:
    """Усиленный мониторинг памяти с более строгими проверками"""

    def __init__(self, threshold=70.0):  # Понижено до 70%
        self.threshold = threshold
        self.monitoring = False
        self.monitor_thread = None
        self.check_count = 0
        self.critical_warnings = 0

    def check_memory(self):
        """Проверка текущего состояния памяти с расширенной диагностикой"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            'percent': memory.percent,
            'available_gb': memory.available / 1024 ** 3,
            'total_gb': memory.total / 1024 ** 3,
            'used_gb': memory.used / 1024 ** 3,
            'swap_percent': swap.percent,
            'swap_used_gb': swap.used / 1024 ** 3
        }

    def force_cleanup(self):
        """Принудительная очистка памяти с более агрессивными методами"""
        logger.info("🧹 Принудительная очистка памяти...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Множественная сборка мусора
        for i in range(5):
            gc.collect()
            time.sleep(0.5)

        # Попытка освободить системную память
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass

        logger.info("   Очистка завершена")

    def emergency_stop_if_needed(self):
        """Усиленная проверка с множественными критериями"""
        memory_info = self.check_memory()
        self.check_count += 1

        # Критические условия
        critical_conditions = []

        if memory_info['percent'] > self.threshold:
            critical_conditions.append(f"RAM: {memory_info['percent']:.1f}%")

        if memory_info['available_gb'] < 2.0:  # Минимум 2GB
            critical_conditions.append(f"Доступно: {memory_info['available_gb']:.1f}GB")

        if memory_info['swap_percent'] > 50:  # Swap не должен активно использоваться
            critical_conditions.append(f"Swap: {memory_info['swap_percent']:.1f}%")

        # CPU проверка
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > CPU_THRESHOLD:
            critical_conditions.append(f"CPU: {cpu_percent:.1f}%")

        if critical_conditions:
            self.critical_warnings += 1
            logger.critical(f"🚨 КРИТИЧЕСКОЕ СОСТОЯНИЕ #{self.critical_warnings}: {', '.join(critical_conditions)}")

            # Принудительная очистка
            self.force_cleanup()

            # Повторная проверка после очистки
            memory_info = self.check_memory()
            if memory_info['percent'] > self.threshold:
                global EMERGENCY_STOP
                EMERGENCY_STOP = True
                raise MemoryError(
                    f"Критическое использование памяти: {memory_info['percent']:.1f}% (порог: {self.threshold}%)")

        # Предупреждения при приближении к лимитам
        if memory_info['percent'] > self.threshold - 10:  # 60% для порога 70%
            logger.warning(f"⚠️ Приближение к лимиту памяти: {memory_info['percent']:.1f}%")

        return memory_info

    def monitor_loop(self):
        """Цикл мониторинга с расширенными проверками"""
        consecutive_warnings = 0

        while self.monitoring:
            try:
                memory_info = self.check_memory()
                cpu_percent = psutil.cpu_percent(interval=1)

                # Детальное логирование каждые 10 проверок
                if self.check_count % 10 == 0:
                    logger.info(f"📊 Система - RAM: {memory_info['percent']:.1f}%, "
                                f"CPU: {cpu_percent:.1f}%, "
                                f"Доступно: {memory_info['available_gb']:.1f}GB, "
                                f"Swap: {memory_info['swap_percent']:.1f}%")

                # Проверка критических условий
                warning_conditions = []

                if memory_info['percent'] > self.threshold - 5:  # 65% для порога 70%
                    warning_conditions.append(f"RAM близко к лимиту: {memory_info['percent']:.1f}%")

                if cpu_percent > 85:
                    warning_conditions.append(f"Высокий CPU: {cpu_percent:.1f}%")

                if memory_info['swap_percent'] > 25:
                    warning_conditions.append(f"Активный Swap: {memory_info['swap_percent']:.1f}%")

                if warning_conditions:
                    consecutive_warnings += 1
                    logger.warning(f"⚠️ Предупреждение #{consecutive_warnings}: {', '.join(warning_conditions)}")

                    if consecutive_warnings >= 5:  # 5 предупреждений подряд
                        logger.error("🚨 Слишком много предупреждений подряд - принудительная очистка")
                        self.force_cleanup()
                        consecutive_warnings = 0
                else:
                    consecutive_warnings = 0

                time.sleep(self.monitor_interval)

            except Exception as e:
                logger.error(f"Ошибка мониторинга: {e}")
                time.sleep(self.monitor_interval)

    def start_monitoring(self):
        """Запуск мониторинга"""
        if not self.monitoring:
            logger.info(f"🔍 Запуск мониторинга памяти (порог: {self.threshold}%)")
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()

    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)


memory_monitor = MemoryMonitor(MEMORY_THRESHOLD)


# ====== DATASET ДЛЯ ОДНОГО ФАЙЛА ======
class SingleFileAMSR2Dataset(Dataset):
    """Dataset для обработки одного NPZ файла"""

    def __init__(self, npz_path: str, preprocessor,
                 degradation_scale: int = 4, augment: bool = True,
                 filter_orbit_type: Optional[str] = None):

        self.npz_path = npz_path
        self.preprocessor = preprocessor
        self.degradation_scale = degradation_scale
        self.augment = augment
        self.filter_orbit_type = filter_orbit_type

        # Загружаем данные из файла
        self.swaths = self._load_file_data()

    def _load_file_data(self):
        """Загрузка данных из одного файла"""
        logger.info(f"📂 Загрузка файла: {os.path.basename(self.npz_path)}")

        try:
            # Проверка памяти перед загрузкой
            memory_monitor.emergency_stop_if_needed()

            with np.load(self.npz_path, allow_pickle=True) as data:
                if 'swath_array' not in data:
                    logger.error(f"❌ Неверная структура файла: {self.npz_path}")
                    return []

                swath_array = data['swath_array']
                valid_swaths = []

                for swath_idx, swath_dict in enumerate(swath_array):
                    swath = swath_dict.item() if isinstance(swath_dict, np.ndarray) else swath_dict

                    # Проверка структуры swath
                    if 'temperature' not in swath or 'metadata' not in swath:
                        logger.warning(f"⚠️ Пропуск swath {swath_idx}: неверная структура")
                        continue

                    # Фильтрация по типу орбиты
                    if self.filter_orbit_type is not None:
                        orbit_type = swath['metadata'].get('orbit_type', 'U')
                        if orbit_type != self.filter_orbit_type:
                            continue

                    # Обработка температуры
                    raw_temperature = swath['temperature']
                    metadata = swath['metadata']

                    # Применение scale factor
                    scale_factor = metadata.get('scale_factor', 1.0)

                    if raw_temperature.dtype != np.float32:
                        temperature = raw_temperature.astype(np.float32) * scale_factor
                    else:
                        temperature = raw_temperature * scale_factor

                    # Фильтрация невалидных значений
                    temperature = np.where(temperature < 50, np.nan, temperature)
                    temperature = np.where(temperature > 350, np.nan, temperature)

                    # Проверка валидности данных
                    valid_pixels = np.sum(~np.isnan(temperature))
                    total_pixels = temperature.size

                    if valid_pixels / total_pixels < 0.1:
                        continue

                    valid_swaths.append({
                        'temperature': temperature,
                        'metadata': metadata
                    })

                logger.info(f"✅ Загружено {len(valid_swaths)} валидных swaths из {len(swath_array)}")
                return valid_swaths

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки файла {self.npz_path}: {e}")
            return []

    def __len__(self):
        return len(self.swaths)

    def __getitem__(self, idx):
        if EMERGENCY_STOP or idx >= len(self.swaths):
            empty_tensor = torch.zeros(1, self.preprocessor.target_height, self.preprocessor.target_width)
            return empty_tensor, empty_tensor

        try:
            swath = self.swaths[idx]
            temperature = swath['temperature'].copy()

            # Обработка данных
            temperature = self.preprocessor.crop_and_pad_to_target(temperature)
            temperature = self.preprocessor.normalize_brightness_temperature(temperature)

            if self.augment:
                temperature = self._augment_data(temperature)

            degraded = self._create_degradation(temperature)

            high_res = torch.from_numpy(temperature).unsqueeze(0).float()
            low_res = torch.from_numpy(degraded).unsqueeze(0).float()

            return low_res, high_res

        except Exception as e:
            logger.error(f"❌ Ошибка в __getitem__[{idx}]: {e}")
            empty_tensor = torch.zeros(1, self.preprocessor.target_height, self.preprocessor.target_width)
            return empty_tensor, empty_tensor

    def _create_degradation(self, high_res: np.ndarray) -> np.ndarray:
        """Создание деградированной версии"""
        try:
            h, w = high_res.shape
            scale = self.degradation_scale

            # Простое downsample
            down_h, down_w = h // scale, w // scale
            if down_h == 0 or down_w == 0:
                return high_res

            cropped = high_res[:down_h * scale, :down_w * scale]
            reshaped = cropped.reshape(down_h, scale, down_w, scale)
            downsampled = np.mean(reshaped, axis=(1, 3))

            # Легкий шум
            noise = np.random.normal(0, 0.01, downsampled.shape).astype(np.float32)
            downsampled = downsampled + noise

            # Upsampling
            upsampled = np.repeat(np.repeat(downsampled, scale, axis=0), scale, axis=1)
            result = upsampled[:h, :w]

            return result

        except Exception as e:
            logger.error(f"Ошибка создания деградации: {e}")
            return high_res

    def _augment_data(self, data: np.ndarray) -> np.ndarray:
        """Легкая аугментация"""
        if not self.augment or np.random.rand() > 0.3:
            return data

        if np.random.rand() > 0.5:
            data = np.fliplr(data)
        if np.random.rand() > 0.5:
            data = np.flipud(data)

        return data


# ====== PREPROCESSOR ======
class AMSR2NPZDataPreprocessor:
    """Препроцессор для AMSR2 данных с правильными размерами"""

    def __init__(self, target_height: int = 2000, target_width: int = 420):
        self.target_height = target_height
        self.target_width = target_width
        logger.info(f"📏 Preprocessor настроен на размер: {target_height}x{target_width}")

    def crop_and_pad_to_target(self, temperature: np.ndarray) -> np.ndarray:
        """Обрезка или дополнение до target размера AMSR2"""
        original_shape = temperature.shape
        h, w = original_shape

        # Обрезка если больше target
        if h > self.target_height:
            start_h = (h - self.target_height) // 2
            temperature = temperature[start_h:start_h + self.target_height]

        if w > self.target_width:
            start_w = (w - self.target_width) // 2
            temperature = temperature[:, start_w:start_w + self.target_width]

        current_h, current_w = temperature.shape

        # Дополнение если меньше target
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
            logger.debug(f"Размер изменен: {original_shape} → {final_shape}")

        return temperature

    def normalize_brightness_temperature(self, temperature: np.ndarray) -> np.ndarray:
        """Нормализация brightness temperature"""
        valid_mask = ~np.isnan(temperature)
        if np.sum(valid_mask) > 0:
            mean_temp = np.mean(temperature[valid_mask])
            temperature = np.where(np.isnan(temperature), mean_temp, temperature)
        else:
            temperature = np.full_like(temperature, 250.0)

        temperature = np.clip(temperature, 50, 350)
        normalized = (temperature - 200) / 150

        return normalized.astype(np.float32)


# ====== АРХИТЕКТУРА МОДЕЛИ (БЕЗ ИЗМЕНЕНИЙ) ======
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


# ====== LOSS И МЕТРИКИ ======
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


# ====== SEQUENTIAL TRAINER ======
class SequentialAMSR2Trainer:
    """Тренер для последовательного обучения на файлах"""

    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        self.criterion = AMSR2SpecificLoss()
        self.training_history = []
        self.best_loss = float('inf')

    def train_on_file(self, file_path: str, preprocessor, batch_size: int = 2,
                      augment: bool = True, filter_orbit_type: Optional[str] = None):
        """Обучение на одном файле"""

        logger.info(f"📚 Обучение на файле: {os.path.basename(file_path)}")

        # Проверка памяти перед началом
        memory_monitor.emergency_stop_if_needed()

        # Создание dataset для одного файла
        dataset = SingleFileAMSR2Dataset(
            npz_path=file_path,
            preprocessor=preprocessor,
            degradation_scale=4,
            augment=augment,
            filter_orbit_type=filter_orbit_type
        )

        if len(dataset) == 0:
            logger.warning(f"⚠️ Пустой файл, пропускаем: {file_path}")
            return {'loss': float('inf'), 'swaths': 0}

        # DataLoader с минимальными настройками
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Без дополнительных процессов
            pin_memory=False
        )

        self.model.train()
        file_losses = []

        try:
            for batch_idx, (low_res, high_res) in enumerate(dataloader):
                if EMERGENCY_STOP:
                    logger.warning("⚠️ Экстренная остановка во время обучения")
                    break

                # Проверка памяти каждые 10 батчей
                if batch_idx % 10 == 0:
                    memory_monitor.emergency_stop_if_needed()

                low_res = low_res.to(self.device)
                high_res = high_res.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(low_res)
                loss, loss_components = self.criterion(pred, high_res)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                file_losses.append(loss.item())

                # Освобождение памяти
                del low_res, high_res, pred, loss

        except Exception as e:
            logger.error(f"❌ Ошибка обучения на файле {file_path}: {e}")
            return {'loss': float('inf'), 'swaths': 0}

        # Очистка памяти после файла
        del dataset, dataloader
        memory_monitor.force_cleanup()

        avg_loss = np.mean(file_losses) if file_losses else float('inf')

        logger.info(f"✅ Файл обработан: loss={avg_loss:.4f}, batches={len(file_losses)}")

        return {'loss': avg_loss, 'swaths': len(dataset)}

    def train_sequential(self, npz_files: List[str], preprocessor,
                         epochs_per_file: int = 1, batch_size: int = 2,
                         save_path: str = "best_amsr2_model.pth"):
        """Последовательное обучение на всех файлах"""

        logger.info(f"🚀 Начинаем последовательное обучение:")
        logger.info(f"   Файлов: {len(npz_files)}")
        logger.info(f"   Эпох на файл: {epochs_per_file}")
        logger.info(f"   Batch size: {batch_size}")

        total_files = len(npz_files)
        processed_files = 0

        for file_idx, file_path in enumerate(npz_files):
            if EMERGENCY_STOP:
                logger.warning("⚠️ Экстренная остановка обучения")
                break

            logger.info(f"\n📂 Файл {file_idx + 1}/{total_files}: {os.path.basename(file_path)}")

            file_results = []

            # Несколько эпох на одном файле
            for epoch in range(epochs_per_file):
                if EMERGENCY_STOP:
                    break

                logger.info(f"   Эпоха {epoch + 1}/{epochs_per_file}")

                result = self.train_on_file(
                    file_path=file_path,
                    preprocessor=preprocessor,
                    batch_size=batch_size,
                    augment=True
                )

                file_results.append(result)

                # Сохранение лучшей модели
                if result['loss'] < self.best_loss:
                    self.best_loss = result['loss']
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.best_loss,
                        'file_idx': file_idx,
                        'epoch': epoch
                    }, save_path)
                    logger.info(f"💾 Сохранена лучшая модель: loss={self.best_loss:.4f}")

            # Статистика по файлу
            avg_file_loss = np.mean([r['loss'] for r in file_results if r['loss'] != float('inf')])
            total_swaths = sum(r['swaths'] for r in file_results)

            self.training_history.append({
                'file_idx': file_idx,
                'file_path': file_path,
                'avg_loss': avg_file_loss,
                'total_swaths': total_swaths,
                'epochs': epochs_per_file
            })

            processed_files += 1

            logger.info(f"📊 Файл завершен: avg_loss={avg_file_loss:.4f}, swaths={total_swaths}")
            logger.info(f"   Прогресс: {processed_files}/{total_files} файлов")

            # Scheduler step
            if avg_file_loss != float('inf'):
                self.scheduler.step(avg_file_loss)

        logger.info(f"\n🎉 Последовательное обучение завершено!")
        logger.info(f"   Обработано файлов: {processed_files}/{total_files}")
        logger.info(f"   Лучший loss: {self.best_loss:.4f}")

        return self.training_history


# ====== ФУНКЦИИ УТИЛИТЫ ======
def find_npz_files(directory: str, max_files: Optional[int] = None) -> List[str]:
    """Поиск NPZ файлов с возможностью ограничения количества"""

    if not os.path.exists(directory):
        logger.error(f"❌ Директория не существует: {directory}")
        return []

    pattern = os.path.join(directory, "*.npz")
    all_files = glob.glob(pattern)

    if not all_files:
        logger.error(f"❌ NPZ файлы не найдены в директории: {directory}")
        return []

    # Сортируем файлы для воспроизводимости
    all_files.sort()

    if max_files is not None and max_files > 0:
        selected_files = all_files[:max_files]
        logger.info(f"📁 Найдено {len(all_files)} NPZ файлов, выбрано {len(selected_files)}")
    else:
        selected_files = all_files
        logger.info(f"📁 Найдено {len(selected_files)} NPZ файлов")

    # Проверка размеров файлов
    total_size_gb = 0
    for file_path in selected_files:
        size_gb = os.path.getsize(file_path) / 1024 ** 3
        total_size_gb += size_gb

    logger.info(f"📊 Общий размер данных: {total_size_gb:.2f} GB")

    if total_size_gb > 10:
        logger.warning(f"⚠️ Большой объем данных! Рекомендуется начать с меньшего количества файлов")

    return selected_files


def validate_npz_structure(file_path: str) -> bool:
    """Проверка структуры NPZ файла"""
    try:
        with np.load(file_path, allow_pickle=True) as data:
            if 'swath_array' not in data:
                return False

            swath_array = data['swath_array']
            if len(swath_array) == 0:
                return False

            # Проверяем первый swath
            first_swath = swath_array[0]
            swath = first_swath.item() if isinstance(first_swath, np.ndarray) else first_swath

            # Проверяем структуру
            required_keys = ['temperature', 'metadata']
            for key in required_keys:
                if key not in swath:
                    return False

            # Проверяем metadata
            metadata = swath['metadata']
            if 'scale_factor' not in metadata:
                logger.warning(f"⚠️ Отсутствует scale_factor в {file_path}")

            return True

    except Exception as e:
        logger.error(f"❌ Ошибка проверки файла {file_path}: {e}")
        return False


def estimate_training_time(num_files: int, avg_swaths_per_file: int = 10,
                           epochs_per_file: int = 1, batch_size: int = 2) -> Dict:
    """Примерная оценка времени обучения"""

    # Базовые оценки (очень приблизительные)
    seconds_per_batch = 2.0  # секунд на batch
    batches_per_file = avg_swaths_per_file // batch_size

    total_batches = num_files * batches_per_file * epochs_per_file
    total_seconds = total_batches * seconds_per_batch

    return {
        'total_batches': total_batches,
        'estimated_hours': total_seconds / 3600,
        'estimated_days': total_seconds / (3600 * 24),
        'batches_per_file': batches_per_file
    }


def create_training_summary(training_history: List[Dict], save_path: str = "training_summary.json"):
    """Создание сводки обучения"""

    if not training_history:
        return

    valid_results = [h for h in training_history if h['avg_loss'] != float('inf')]

    if not valid_results:
        logger.warning("⚠️ Нет валидных результатов для сводки")
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

    logger.info(f"📄 Сводка сохранена: {save_path}")
    logger.info(f"   Успешных файлов: {summary['successful_files']}/{summary['total_files_processed']}")
    logger.info(f"   Общее количество swaths: {summary['total_swaths']}")
    logger.info(f"   Лучший loss: {summary['best_loss']:.4f}")


def plot_training_progress(training_history: List[Dict], save_path: str = "training_progress.png"):
    """Визуализация прогресса обучения"""

    if not training_history:
        return

    valid_results = [h for h in training_history if h['avg_loss'] != float('inf')]

    if len(valid_results) < 2:
        logger.warning("⚠️ Недостаточно данных для графика")
        return

    file_indices = [h['file_idx'] for h in valid_results]
    losses = [h['avg_loss'] for h in valid_results]
    swaths = [h['total_swaths'] for h in valid_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # График loss
    ax1.plot(file_indices, losses, 'b-o', linewidth=2, markersize=6)
    ax1.set_title('Training Loss по файлам', fontsize=14)
    ax1.set_xlabel('Номер файла')
    ax1.set_ylabel('Average Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # График количества swaths
    ax2.bar(file_indices, swaths, alpha=0.7, color='green')
    ax2.set_title('Количество swaths в файлах', fontsize=14)
    ax2.set_xlabel('Номер файла')
    ax2.set_ylabel('Количество swaths')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"📈 График сохранен: {save_path}")


# ====== MAIN ФУНКЦИЯ ======
def main():
    """Главная функция для последовательного обучения"""

    parser = argparse.ArgumentParser(
        description='AMSR2 Sequential Super-Resolution Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:

1. Обучение на первых 10 файлах:
   python sequential_amsr2.py --npz-dir /path/to/data --max-files 10

2. Обучение на всех файлах с фильтром орбиты:
   python sequential_amsr2.py --npz-dir /path/to/data --orbit-filter A

3. Быстрое тестирование на 5 файлах:
   python sequential_amsr2.py --npz-dir /path/to/data --max-files 5 --epochs-per-file 1 --batch-size 1
        '''
    )

    # Обязательные параметры
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='Путь к директории с NPZ файлами')

    # Параметры данных
    parser.add_argument('--max-files', type=int, default=None,
                        help='Максимальное количество файлов для обучения (по умолчанию: все)')
    parser.add_argument('--orbit-filter', choices=['A', 'D', 'U'],
                        help='Фильтр по типу орбиты (A=Ascending, D=Descending, U=Unknown)')

    # Параметры обучения
    parser.add_argument('--epochs-per-file', type=int, default=2,
                        help='Количество эпох обучения на каждом файле (по умолчанию: 2)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Размер batch (по умолчанию: 2)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (по умолчанию: 1e-4)')
    parser.add_argument('--scale-factor', type=int, default=10,
                        help='Фактор увеличения разрешения (по умолчанию: 10)')

    # Параметры системы с пониженными порогами
    parser.add_argument('--memory-threshold', type=float, default=70.0,
                        help='Порог использования памяти для экстренной остановки (по умолчанию: 70%%)')
    parser.add_argument('--target-height', type=int, default=2000,
                        help='Целевая высота изображений для AMSR2 (по умолчанию: 2000)')
    parser.add_argument('--target-width', type=int, default=420,
                        help='Целевая ширина изображений для AMSR2 (по умолчанию: 420)')

    # Пути сохранения
    parser.add_argument('--save-path', type=str, default='best_amsr2_sequential_model.pth',
                        help='Путь для сохранения лучшей модели')
    parser.add_argument('--no-monitoring', action='store_true',
                        help='Отключить мониторинг памяти')

    args = parser.parse_args()

    print("🛰️ AMSR2 SEQUENTIAL SUPER-RESOLUTION TRAINER")
    print("=" * 60)

    # Проверка доступности GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ Используется устройство: {device}")

    # Проверка памяти с более строгими требованиями
    memory_info = psutil.virtual_memory()
    logger.info(
        f"💾 Память: {memory_info.available / 1024 ** 3:.1f} GB доступно из {memory_info.total / 1024 ** 3:.1f} GB ({memory_info.percent:.1f}% занято)")

    if memory_info.percent > 60:  # Предупреждение уже при 60%
        logger.warning(f"⚠️ Высокое использование памяти: {memory_info.percent:.1f}%")
        logger.warning("   Рекомендуется освободить память перед обучением")

    if memory_info.available < 4 * 1024 ** 3:  # Минимум 4GB
        logger.error(f"❌ Недостаточно доступной памяти: {memory_info.available / 1024 ** 3:.1f} GB")
        logger.error("   Необходимо минимум 4 GB для безопасного обучения")
        sys.exit(1)

    # Настройка мониторинга
    global memory_monitor
    memory_monitor = MemoryMonitor(args.memory_threshold)

    if not args.no_monitoring:
        memory_monitor.start_monitoring()

    try:
        # Поиск NPZ файлов
        npz_files = find_npz_files(args.npz_dir, args.max_files)

        if not npz_files:
            logger.error("❌ Не найдено подходящих NPZ файлов")
            sys.exit(1)

        # Валидация структуры файлов (проверяем первые несколько)
        logger.info("🔍 Проверка структуры файлов...")
        valid_files = []

        for i, file_path in enumerate(npz_files[:min(5, len(npz_files))]):
            if validate_npz_structure(file_path):
                logger.info(f"✅ Файл {i + 1} валиден: {os.path.basename(file_path)}")
            else:
                logger.error(f"❌ Файл {i + 1} невалиден: {os.path.basename(file_path)}")

        # Оценка времени обучения
        time_estimate = estimate_training_time(
            len(npz_files),
            epochs_per_file=args.epochs_per_file,
            batch_size=args.batch_size
        )

        logger.info(f"⏱️ Примерная оценка времени обучения:")
        logger.info(f"   Общее количество батчей: {time_estimate['total_batches']}")
        logger.info(
            f"   Примерное время: {time_estimate['estimated_hours']:.1f} часов ({time_estimate['estimated_days']:.1f} дней)")

        if time_estimate['estimated_hours'] > 48:
            logger.warning("⚠️ Обучение займет очень много времени!")
            logger.warning("   Рекомендуется уменьшить --max-files или --epochs-per-file")

        # Создание модели
        logger.info("🧠 Создание модели...")
        model = UNetResNetSuperResolution(
            in_channels=1,
            out_channels=1,
            scale_factor=args.scale_factor
        )

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Параметров в модели: {total_params:,}")
        logger.info(f"   Фактор увеличения: {args.scale_factor}x")

        # Создание preprocessor
        preprocessor = AMSR2NPZDataPreprocessor(
            target_height=args.target_size,
            target_width=args.target_size
        )

        # Создание тренера
        trainer = SequentialAMSR2Trainer(
            model=model,
            device=device,
            learning_rate=args.lr
        )

        # Вывод конфигурации
        logger.info(f"⚙️ Конфигурация обучения:")
        logger.info(f"   Файлов для обучения: {len(npz_files)}")
        logger.info(f"   Эпох на файл: {args.epochs_per_file}")
        logger.info(f"   Batch size: {args.batch_size}")
        logger.info(f"   Learning rate: {args.lr}")
        logger.info(f"   Target размеры: {args.target_height}x{args.target_width}")
        logger.info(f"   Порог памяти: {args.memory_threshold}%")

        if args.orbit_filter:
            logger.info(f"   Фильтр орбиты: {args.orbit_filter}")

        # Подтверждение от пользователя
        try:
            confirm = input("\n🚀 Начать обучение? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes', 'да', 'д']:
                logger.info("❌ Обучение отменено пользователем")
                sys.exit(0)
        except KeyboardInterrupt:
            logger.info("\n❌ Обучение отменено пользователем")
            sys.exit(0)

        # Запуск обучения
        logger.info("\n🚀 Запуск последовательного обучения...")
        start_time = time.time()

        training_history = trainer.train_sequential(
            npz_files=npz_files,
            preprocessor=preprocessor,
            epochs_per_file=args.epochs_per_file,
            batch_size=args.batch_size,
            save_path=args.save_path
        )

        end_time = time.time()
        training_time = end_time - start_time

        logger.info(f"\n🎉 Обучение завершено!")
        logger.info(f"   Время обучения: {training_time / 3600:.2f} часов")
        logger.info(f"   Файлов обработано: {len(training_history)}")

        # Создание отчетов
        create_training_summary(training_history)
        plot_training_progress(training_history)

        # Финальная статистика памяти
        if not args.no_monitoring:
            stats_summary = memory_monitor.get_stats_summary()
            if stats_summary:
                logger.info(f"📊 Статистика памяти:")
                logger.info(f"   Средняя загрузка: {stats_summary.get('avg_memory_percent', 0):.1f}%")
                logger.info(f"   Максимальная загрузка: {stats_summary.get('max_memory_percent', 0):.1f}%")
                logger.info(f"   Минимум доступной памяти: {stats_summary.get('min_available_memory_gb', 0):.1f} GB")

        logger.info(f"\n📁 Результаты:")
        logger.info(f"   Модель: {args.save_path}")
        logger.info(f"   Сводка: training_summary.json")
        logger.info(f"   График: training_progress.png")
        logger.info(f"   Лог: amsr2_sequential.log")

    except KeyboardInterrupt:
        logger.info("\n⏹️ Обучение прервано пользователем")
    except MemoryError as e:
        logger.critical(f"\n💾 Критическая ошибка памяти: {e}")
    except Exception as e:
        logger.error(f"\n❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Остановка мониторинга
        if not args.no_monitoring:
            memory_monitor.stop_monitoring()

        logger.info("🛑 Завершение работы программы")


if __name__ == "__main__":
    main()