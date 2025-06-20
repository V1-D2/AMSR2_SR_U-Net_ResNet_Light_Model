#!/usr/bin/env python3
"""
Простой launcher для AMSR2 Super-Resolution
Обеспечивает удобный интерфейс для запуска полной версии
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_dependencies():
    """Проверка установленных пакетов"""

    required_packages = [
        'torch', 'torchvision', 'numpy', 'matplotlib',
        'scikit-learn', 'tqdm'
    ]

    missing = []

    print("🔍 Проверка зависимостей...")

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")

    if missing:
        print(f"\n❌ Отсутствуют пакеты: {missing}")
        print("Установите командой:")
        print(f"pip install {' '.join(missing)}")
        return False

    print("✅ Все зависимости установлены")
    return True


def check_system():
    """Проверка системы"""

    print("\n🖥️ Проверка системы...")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")

            if gpu_memory < 6:
                print("⚠️ Мало GPU памяти. Используйте --batch-size 4")
        else:
            print("⚠️ CUDA недоступна, будет использован CPU")
            print("⚠️ Обучение на CPU будет очень медленным")
    except ImportError:
        print("❌ PyTorch не установлен")
        return False

    return True


def analyze_npz_directory(npz_dir):
    """Быстрый анализ NPZ директории"""

    if not os.path.exists(npz_dir):
        return False, f"Директория не найдена: {npz_dir}"

    import glob
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))

    if not npz_files:
        return False, f"NPZ файлы не найдены в {npz_dir}"

    print(f"📁 Найдено {len(npz_files)} NPZ файлов в {npz_dir}")

    # Проверим первый файл
    try:
        import numpy as np
        with np.load(npz_files[0], allow_pickle=True) as data:
            if 'swath_array' in data:
                swath_array = data['swath_array']
                print(f"✅ Формат корректен: {len(swath_array)} swaths в первом файле")
                return True, f"Готово к обучению: {len(npz_files)} файлов"
            else:
                return False, "Неверный формат NPZ файла"
    except Exception as e:
        return False, f"Ошибка чтения NPZ: {e}"


def main():
    """Главная функция launcher"""

    parser = argparse.ArgumentParser(
        description="Простой launcher для AMSR2 Super-Resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

1. Быстрый тест с синтетическими данными:
   python run_amsr2_simple.py --test

2. Обучение на ваших NPZ данных:
   python run_amsr2_simple.py --train --npz-dir /path/to/your/npz/files

3. Обучение с настройками:
   python run_amsr2_simple.py --train --npz-dir /path/to/data \\
       --batch-size 4 --epochs 30 --lr 2e-4

4. Inference на новом файле:
   python run_amsr2_simple.py --inference \\
       --model best_amsr2_model.pth \\
       --input data.npz \\
       --output enhanced.npz
        """
    )

    # Режимы работы
    parser.add_argument('--test', action='store_true',
                        help='Быстрый тест с синтетическими данными')
    parser.add_argument('--train', action='store_true',
                        help='Обучение модели')
    parser.add_argument('--inference', action='store_true',
                        help='Inference на новых данных')

    # Параметры данных
    parser.add_argument('--npz-dir', type=str, default='./npz_data',
                        help='Путь к папке с NPZ файлами')

    # Параметры обучения
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Размер batch (уменьшите если мало GPU памяти)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Количество эпох')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--orbit-filter', choices=['A', 'D', 'U'],
                        help='Фильтр по типу орбиты')

    # Параметры inference
    parser.add_argument('--model', type=str, default='best_amsr2_model.pth',
                        help='Путь к файлу модели')
    parser.add_argument('--input', type=str,
                        help='Входной NPZ файл')
    parser.add_argument('--output', type=str,
                        help='Выходной файл результата')

    args = parser.parse_args()

    print("🛰️ AMSR2 BRIGHTNESS TEMPERATURE SUPER-RESOLUTION")
    print("=" * 60)

    # Проверка системы
    if not check_dependencies():
        sys.exit(1)

    if not check_system():
        sys.exit(1)

    # Определение режима работы
    if args.test:
        print("\n🚀 РЕЖИМ: Быстрое тестирование")
        cmd = [
            sys.executable, "amsr2_super_resolution_complete.py",
            "--test-mode",
            "--epochs", "5",
            "--batch-size", str(min(args.batch_size, 4))
        ]

    elif args.train:
        print(f"\n🎯 РЕЖИМ: Обучение на данных из {args.npz_dir}")

        # Анализ данных
        success, message = analyze_npz_directory(args.npz_dir)
        print(message)

        if not success:
            print("\n❌ Проблема с данными. Запускаем тестовый режим...")
            args.test = True
            cmd = [
                sys.executable, "amsr2_super_resolution_complete.py",
                "--test-mode",
                "--epochs", "5"
            ]
        else:
            cmd = [
                sys.executable, "amsr2_super_resolution_complete.py",
                "--npz-dir", args.npz_dir,
                "--batch-size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--lr", str(args.lr)
            ]

            if args.orbit_filter:
                cmd.extend(["--orbit-filter", args.orbit_filter])

    elif args.inference:
        print("\n🔮 РЕЖИМ: Inference")

        if not args.input or not args.output:
            print("❌ Для inference требуются --input и --output")
            sys.exit(1)

        if not os.path.exists(args.model):
            print(f"❌ Модель не найдена: {args.model}")
            sys.exit(1)

        if not os.path.exists(args.input):
            print(f"❌ Входной файл не найден: {args.input}")
            sys.exit(1)

        # Для inference используем импорт функции
        try:
            from amsr2_super_resolution_complete import process_new_npz_file

            enhanced_swaths = process_new_npz_file(
                model_path=args.model,
                npz_path=args.input,
                output_path=args.output
            )

            print(f"✅ Обработано {len(enhanced_swaths)} swaths")
            print(f"💾 Результат: {args.output}")
            return

        except Exception as e:
            print(f"❌ Ошибка inference: {e}")
            sys.exit(1)

    else:
        print("\n❌ Выберите режим: --test, --train или --inference")
        parser.print_help()
        sys.exit(1)

    # Запуск основной программы
    print(f"\n🚀 Запуск команды:")
    print(" ".join(cmd))
    print("\n" + "=" * 60)

    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("🎉 ВЫПОЛНЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")

        if args.test or args.train:
            print("\n📁 Проверьте созданные файлы:")
            print("  - best_amsr2_model.pth (веса модели)")
            print("  - amsr2_training_history.png (графики)")
            print("  - amsr2_training_summary.json (статистика)")

    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("❌ ОШИБКА ВЫПОЛНЕНИЯ")
        print(f"Код возврата: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Выполнение прервано пользователем")
        sys.exit(1)


if __name__ == "__main__":
    main()