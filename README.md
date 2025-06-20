# AMSR2 Brightness Temperature Super-Resolution

🛰️ **Увеличение разрешения данных AMSR2 в 10 раз с помощью глубокого обучения**

Этот проект реализует super-resolution модель для данных brightness temperature от спутника AMSR2, увеличивая пространственное разрешение с ~10км/пиксель до ~1км/пиксель (увеличение в 10 раз).

## 🎯 Ключевые особенности

- **Self-supervised learning**: Обучение без парных данных высокого разрешения
- **NPZ формат**: Работает напрямую с NPZ файлами вашего процессора
- **U-Net + ResNet**: Проверенная архитектура для scientific imaging
- **Физическая корректность**: Специализированные loss функции для brightness temperature
- **Множественные swath**: Автоматическая обработка всех swath в NPZ файле
- **Orbit filtering**: Фильтрация по типам орбит (Ascending/Descending/Unknown)

## 📁 Структура проекта

```
amsr2-super-resolution/
├── amsr2_super_resolution_complete.py  # 🎯 ОСНОВНОЙ ФАЙЛ (вся модель)
├── run_amsr2_simple.py                 # 🚀 Простой launcher (опционально)
├── README.md                           # 📚 Документация
├── requirements.txt                    # 📦 Зависимости
└── examples/                           # 📊 Примеры результатов
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm
```

### 2. Быстрый тест

```bash
# Вариант 1: Прямой запуск основного файла
python amsr2_super_resolution_complete.py --test-mode

# Вариант 2: Через launcher
python run_amsr2_simple.py --test
```

### 3. Обучение на ваших данных

```bash
# Замените /path/to/your/npz/files на путь к вашим NPZ файлам
python amsr2_super_resolution_complete.py --npz-dir /path/to/your/npz/files

# Или через launcher
python run_amsr2_simple.py --train --npz-dir /path/to/your/npz/files
```

## 📊 Формат входных данных

Модель работает с NPZ файлами в формате вашего AMSR2 процессора:

```python
# Структура NPZ файла:
{
    'swath_array': [
        {
            'temperature': np.array([2041, 421], dtype=uint16),  # Raw данные
            'metadata': {
                'orbit_type': 'A',        # A/D/U
                'scale_factor': 0.1,      # Для конвертации в Kelvin
                'temp_range': (150, 300), # После scaling
                'shape': (2041, 421)
            }
        },
        # ... больше swath
    ],
    'period': '2024-01-01T00:00:00 to 2024-01-01T23:59:59'
}
```

## 🎛️ Параметры запуска

### Основные параметры:

```bash
python amsr2_super_resolution_complete.py \
    --npz-dir /path/to/npz/files \     # Папка с NPZ файлами
    --batch-size 8 \                   # Размер batch (уменьшите если мало GPU)
    --epochs 50 \                      # Количество эпох
    --lr 1e-4 \                        # Learning rate
    --scale-factor 10 \                # Масштаб увеличения
    --orbit-filter A                   # Фильтр по орбитам (A/D/U)
```

### Через launcher:

```bash
python run_amsr2_simple.py --train \
    --npz-dir /path/to/data \
    --batch-size 4 \
    --epochs 30 \
    --orbit-filter A
```

## 🔮 Inference на новых данных

```bash
# Прямой вызов функции
python -c "
from amsr2_super_resolution_complete import process_new_npz_file
process_new_npz_file(
    model_path='best_amsr2_model.pth',
    npz_path='new_data.npz', 
    output_path='enhanced_result.npz'
)"

# Через launcher
python run_amsr2_simple.py --inference \
    --model best_amsr2_model.pth \
    --input new_data.npz \
    --output enhanced_result.npz
```

## 📈 Ожидаемые результаты

- **Входной размер**: 2041×421 → **Выходной размер**: 20410×4210
- **PSNR**: 28-35 dB (отлично для научных данных)
- **SSIM**: 0.7-0.9 (высокая структурная точность)
- **Время обучения**: 
  - 1000 swaths: ~4-6 часов на RTX 3080
  - 5000 swaths: ~20-25 часов на RTX 3080

## 🖥️ Системные требования

### Минимальные:
- **GPU**: 8GB VRAM (RTX 3070/V100)
- **RAM**: 16GB системной памяти
- **Storage**: 50GB свободного места

### Рекомендуемые:
- **GPU**: 16GB+ VRAM (RTX 4090/A100)
- **RAM**: 32GB+ системной памяти
- **Storage**: 100GB+ SSD

## 📁 Выходные файлы

После обучения создаются:

```
📁 Результаты обучения:
├── best_amsr2_model.pth           # 🏆 Веса обученной модели
├── amsr2_training_history.png     # 📊 Графики метрик
├── amsr2_training_summary.json    # 📋 Детальная статистика
├── amsr2_example_1.png            # 🖼️ Примеры результатов
├── amsr2_example_2.png
└── ...
```

## 🛠️ Решение проблем

### "CUDA out of memory"
```bash
# Уменьшите batch size
python amsr2_super_resolution_complete.py --batch-size 4  # вместо 8
```

### "No NPZ files found"
```bash
# Проверьте путь к данным
ls /path/to/your/npz/files/*.npz

# Или запустите тест
python amsr2_super_resolution_complete.py --test-mode
```

### Медленная конвергенция
```bash
# Увеличьте learning rate
python amsr2_super_resolution_complete.py --lr 2e-4  # вместо 1e-4
```

## 🔬 Научная корректность

Модель включает специальные механизмы для сохранения физических свойств:

- **Energy Conservation**: Сохранение общей энергии brightness temperature
- **Gradient Preservation**: Сохранение пространственных структур
- **Range Validation**: Контроль реалистичности температурных значений
- **Scale Factor Handling**: Корректная обработка raw данных от AMSR2

## 📚 Техническая документация

### Архитектура модели:
- **Encoder**: ResNet-34 backbone с skip connections
- **Decoder**: Progressive upsampling с feature fusion
- **Loss Function**: L1 + Gradient + Physical consistency losses
- **Scale Factor**: Progressive upsampling для 10x увеличения

### Препроцессинг:
1. Загрузка raw данных (uint16) из NPZ
2. Применение scale factor (обычно 0.1)
3. Фильтрация невалидных значений (< 50K, > 350K)
4. Нормализация в диапазон [-1, 1]
5. Создание degraded версии для self-supervised learning

## 📞 Поддержка

При возникновении проблем:

1. Проверьте системные требования
2. Убедитесь в корректности формата NPZ файлов
3. Попробуйте запустить тест: `--test-mode`
4. Проверьте логи в консоли для диагностики

## 📄 Лицензия

MIT License - свободное использование для исследовательских целей.

---

**Разработано для научного сообщества AMSR2** 🛰️