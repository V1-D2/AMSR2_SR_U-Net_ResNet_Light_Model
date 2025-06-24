#!/usr/bin/env python3
"""
Простой launcher для AMSR2 Sequential Trainer
Без CPU оптимизаций, с защитой памяти и возможностью выбора количества файлов
"""

import os
import sys
import subprocess
import argparse
import psutil
import glob
from pathlib import Path


def check_memory():
    """Проверка доступной памяти с более строгими требованиями"""
    memory = psutil.virtual_memory()
    print(f"💾 Память:")
    print(f"   Общая: {memory.total / 1024 ** 3:.1f} GB")
    print(f"   Доступная: {memory.available / 1024 ** 3:.1f} GB")
    print(f"   Используется: {memory.percent:.1f}%")

    # Более строгие проверки
    if memory.percent > 70:  # Понижено с 80% до 70%
        print("🚨 КРИТИЧЕСКОЕ: Память сильно загружена!")
        return False

    if memory.percent > 60:  # Предупреждение при 60%
        print("⚠️ ВНИМАНИЕ: Высокое использование памяти!")
        print("   Рекомендуется освободить память перед обучением")

    if memory.available < 4 * 1024 ** 3:  # Увеличено с 2GB до 4GB
        print("🚨 КРИТИЧЕСКОЕ: Недостаточно доступной памяти!")
        print(f"   Доступно: {memory.available / 1024 ** 3:.1f} GB, требуется минимум 4 GB")
        return False

    return True


def analyze_npz_directory(npz_dir):
    """Анализ директории с NPZ файлами"""
    if not os.path.exists(npz_dir):
        print(f"❌ Директория не существует: {npz_dir}")
        return None

    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))

    if not npz_files:
        print(f"❌ NPZ файлы не найдены в: {npz_dir}")
        return None

    # Сортируем и анализируем
    npz_files.sort()
    total_size_gb = sum(os.path.getsize(f) / 1024 ** 3 for f in npz_files)

    analysis = {
        'total_files': len(npz_files),
        'total_size_gb': total_size_gb,
        'avg_size_mb': (total_size_gb * 1024) / len(npz_files),
        'files': npz_files
    }

    print(f"📁 Анализ директории:")
    print(f"   Найдено файлов: {analysis['total_files']}")
    print(f"   Общий размер: {analysis['total_size_gb']:.2f} GB")
    print(f"   Средний размер файла: {analysis['avg_size_mb']:.1f} MB")

    return analysis


def recommend_parameters(analysis, available_memory_gb):
    """Рекомендация параметров с учетом пониженных порогов памяти"""

    if analysis is None:
        return None

    total_files = analysis['total_files']
    total_size_gb = analysis['total_size_gb']

    # Более консервативные рекомендации
    if available_memory_gb < 6:  # Увеличено с 4GB
        # Мало памяти
        rec = {
            'max_files': min(3, total_files),  # Уменьшено с 5
            'batch_size': 1,
            'epochs_per_file': 1,
            'memory_threshold': 65.0,  # Понижено с 70%
            'description': 'Мало памяти - минимальные параметры'
        }
    elif available_memory_gb < 10:  # Увеличено с 8GB
        # Средняя память
        rec = {
            'max_files': min(8, total_files),  # Уменьшено с 10
            'batch_size': 1,  # Уменьшено с 2
            'epochs_per_file': 2,
            'memory_threshold': 70.0,  # Понижено с 75%
            'description': 'Средняя память - осторожные параметры'
        }
    elif available_memory_gb < 20:  # Увеличено с 16GB
        # Хорошая память
        if total_size_gb < 10:
            rec = {
                'max_files': min(15, total_files),  # Уменьшено с 20
                'batch_size': 2,  # Уменьшено с 3
                'epochs_per_file': 2,
                'memory_threshold': 70.0,
                'description': 'Хорошая память - умеренные параметры'
            }
        else:
            rec = {
                'max_files': min(10, total_files),  # Уменьшено с 15
                'batch_size': 1,  # Уменьшено с 2
                'epochs_per_file': 2,
                'memory_threshold': 70.0,
                'description': 'Хорошая память + большие файлы'
            }
    else:
        # Много памяти
        if total_size_gb < 20:
            rec = {
                'max_files': min(30, total_files),  # Уменьшено с 50
                'batch_size': 3,  # Уменьшено с 4
                'epochs_per_file': 3,
                'memory_threshold': 70.0,  # Понижено с 85%
                'description': 'Много памяти - оптимальные параметры'
            }
        else:
            rec = {
                'max_files': min(20, total_files),  # Уменьшено с 30
                'batch_size': 2,  # Уменьшено с 3
                'epochs_per_file': 2,
                'memory_threshold': 70.0,
                'description': 'Много памяти + очень большие файлы'
            }

    return rec


def estimate_time(max_files, epochs_per_file, batch_size):
    """Примерная оценка времени"""
    # Очень грубая оценка
    avg_swaths_per_file = 10
    seconds_per_batch = 3.0

    batches_per_file = avg_swaths_per_file // batch_size
    total_batches = max_files * batches_per_file * epochs_per_file
    total_seconds = total_batches * seconds_per_batch

    hours = total_seconds / 3600
    days = hours / 24

    return {
        'hours': hours,
        'days': days,
        'total_batches': total_batches
    }


def main():
    parser = argparse.ArgumentParser(
        description='Простой launcher для AMSR2 Sequential Trainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:

1. Быстрый тест на 5 файлах:
   python simple_launcher.py --npz-dir /path/to/data --quick-test

2. Средний тест на 20 файлах:
   python simple_launcher.py --npz-dir /path/to/data --medium-test

3. Полное обучение (осторожно!):
   python simple_launcher.py --npz-dir /path/to/data --full-train

4. Кастомные параметры:
   python simple_launcher.py --npz-dir /path/to/data --max-files 15 --batch-size 2 --epochs 3
        '''
    )

    # Основные параметры
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='Путь к директории с NPZ файлами')

    # Предустановленные режимы
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--quick-test', action='store_true',
                       help='Быстрый тест (5 файлов, 1 эпоха)')
    group.add_argument('--medium-test', action='store_true',
                       help='Средний тест (автоматический выбор параметров)')
    group.add_argument('--full-train', action='store_true',
                       help='Полное обучение (все файлы)')

    # Кастомные параметры
    parser.add_argument('--max-files', type=int,
                        help='Максимальное количество файлов')
    parser.add_argument('--batch-size', type=int,
                        help='Размер batch')
    parser.add_argument('--epochs', type=int,
                        help='Эпох на файл')
    parser.add_argument('--memory-threshold', type=float, default=70.0,
                        help='Порог памяти для экстренной остановки (по умолчанию: 70)')

    # Дополнительные параметры с новыми размерами
    parser.add_argument('--orbit-filter', choices=['A', 'D', 'U'],
                        help='Фильтр по типу орбиты')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--target-height', type=int, default=2000,
                        help='Целевая высота изображений для AMSR2 (по умолчанию: 2000)')
    parser.add_argument('--target-width', type=int, default=420,
                        help='Целевая ширина изображений для AMSR2 (по умолчанию: 420)')
    parser.add_argument('--no-monitoring', action='store_true',
                        help='Отключить мониторинг памяти')

    args = parser.parse_args()

    print("🛰️ AMSR2 SEQUENTIAL TRAINER LAUNCHER")
    print("=" * 50)

    # Проверка памяти
    if not check_memory():
        print("❌ Недостаточно памяти для безопасного обучения")
        print("   Закройте другие программы и попробуйте снова")
        sys.exit(1)

    memory = psutil.virtual_memory()
    available_memory_gb = memory.available / 1024 ** 3

    # Анализ данных
    print(f"\n🔍 Анализ данных в: {args.npz_dir}")
    analysis = analyze_npz_directory(args.npz_dir)

    if analysis is None:
        print("❌ Не удалось найти NPZ файлы")
        print("   Проверьте путь к директории")
        sys.exit(1)

    # Рекомендации
    recommendations = recommend_parameters(analysis, available_memory_gb)
    print(f"\n💡 Рекомендации: {recommendations['description']}")
    print(f"   Макс. файлов: {recommendations['max_files']}")
    print(f"   Batch size: {recommendations['batch_size']}")
    print(f"   Эпох на файл: {recommendations['epochs_per_file']}")

    # Определение финальных параметров
    if args.quick_test:
        final_params = {
            'max_files': 5,
            'batch_size': 1,
            'epochs_per_file': 1,
            'memory_threshold': 75.0,
            'mode': 'Быстрый тест'
        }
    elif args.medium_test:
        final_params = {
            'max_files': recommendations['max_files'],
            'batch_size': recommendations['batch_size'],
            'epochs_per_file': recommendations['epochs_per_file'],
            'memory_threshold': recommendations['memory_threshold'],
            'mode': 'Средний тест'
        }
    elif args.full_train:
        final_params = {
            'max_files': analysis['total_files'],
            'batch_size': max(1, recommendations['batch_size'] - 1),  # Еще более осторожно
            'epochs_per_file': 2,
            'memory_threshold': 70.0,  # Понижено с 80%
            'mode': 'Полное обучение'
        }
    else:
        # Кастомные параметры
        final_params = {
            'max_files': args.max_files or recommendations['max_files'],
            'batch_size': args.batch_size or recommendations['batch_size'],
            'epochs_per_file': args.epochs or recommendations['epochs_per_file'],
            'memory_threshold': args.memory_threshold,
            'mode': 'Кастомные параметры'
        }

    # Более строгие безопасные ограничения
    final_params['max_files'] = min(final_params['max_files'], analysis['total_files'])
    final_params['batch_size'] = max(1, min(final_params['batch_size'], 4))  # Максимум 4 (было 8)
    final_params['epochs_per_file'] = max(1, min(final_params['epochs_per_file'], 3))  # Максимум 3 (было 5)

    # Дополнительные проверки безопасности
    memory_gb = psutil.virtual_memory().available / 1024 ** 3
    if memory_gb < 8 and final_params['batch_size'] > 1:
        logger.warning(f"⚠️ При {memory_gb:.1f}GB памяти снижаем batch_size до 1")
        final_params['batch_size'] = 1

    if analysis['total_size_gb'] > 30 and final_params['max_files'] > 10:
        logger.warning(f"⚠️ При {analysis['total_size_gb']:.1f}GB данных ограничиваем файлы до 10")
        final_params['max_files'] = 10

    # Оценка времени
    time_estimate = estimate_time(
        final_params['max_files'],
        final_params['epochs_per_file'],
        final_params['batch_size']
    )

    print(f"\n⚙️ Финальная конфигурация ({final_params['mode']}):")
    print(f"   Файлов для обучения: {final_params['max_files']} из {analysis['total_files']}")
    print(f"   Batch size: {final_params['batch_size']}")
    print(f"   Эпох на файл: {final_params['epochs_per_file']}")
    print(f"   Порог памяти: {final_params['memory_threshold']}%")

    print(f"\n⏱️ Примерная оценка времени:")
    print(f"   Батчей всего: {time_estimate['total_batches']}")
    print(f"   Время: {time_estimate['hours']:.1f} часов ({time_estimate['days']:.1f} дней)")

    if time_estimate['hours'] > 24:
        print("⚠️ ВНИМАНИЕ: Обучение займет больше суток!")
        print("   Рекомендуется начать с меньшего количества файлов")

    # Предупреждения
    if final_params['max_files'] > 50:
        print("⚠️ ВНИМАНИЕ: Очень много файлов для обучения!")

    if analysis['total_size_gb'] > 30:
        print(f"⚠️ ВНИМАНИЕ: Большой объем данных ({analysis['total_size_gb']:.1f} GB)!")

    # Подтверждение
    print(f"\n🚀 Готовы начать обучение?")
    try:
        confirm = input("Введите 'yes' для продолжения: ").strip().lower()
        if confirm not in ['yes', 'y', 'да', 'д']:
            print("❌ Обучение отменено")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n❌ Обучение отменено")
        sys.exit(0)

    # Построение команды
    cmd = [sys.executable, "sequential_amsr2_trainer.py"]

    # Обязательные параметры
    cmd.extend(["--npz-dir", args.npz_dir])
    cmd.extend(["--max-files", str(final_params['max_files'])])
    cmd.extend(["--batch-size", str(final_params['batch_size'])])
    cmd.extend(["--epochs-per-file", str(final_params['epochs_per_file'])])
    cmd.extend(["--memory-threshold", str(final_params['memory_threshold'])])

    # Дополнительные параметры
    if args.orbit_filter:
        cmd.extend(["--orbit-filter", args.orbit_filter])

    if args.lr != 1e-4:
        cmd.extend(["--lr", str(args.lr)])

    if args.target_height != 2000:
        cmd.extend(["--target-height", str(args.target_height)])

    if args.target_width != 420:
        cmd.extend(["--target-width", str(args.target_width)])

    if args.no_monitoring:
        cmd.append("--no-monitoring")

    # Имя модели в зависимости от режима
    model_name = f"amsr2_model_{final_params['mode'].lower().replace(' ', '_')}_{final_params['max_files']}files.pth"
    cmd.extend(["--save-path", model_name])

    print(f"\n🔧 Команда запуска:")
    print(" ".join(cmd))
    print(f"\n📁 Модель будет сохранена как: {model_name}")

    print("\n" + "=" * 50)
    print("🚀 ЗАПУСК ОБУЧЕНИЯ")
    print("=" * 50)

    try:
        # Запуск
        result = subprocess.run(cmd, check=True)

        print("\n" + "=" * 50)
        print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 50)
        print(f"📁 Результаты:")
        print(f"   Модель: {model_name}")
        print(f"   Сводка: training_summary.json")
        print(f"   График: training_progress.png")
        print(f"   Лог: amsr2_sequential.log")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ ОШИБКА ВЫПОЛНЕНИЯ (код: {e.returncode})")
        print("Проверьте лог amsr2_sequential.log для диагностики")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️ Обучение прервано пользователем")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()