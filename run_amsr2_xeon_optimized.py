#!/usr/bin/env python3
"""
Оптимизированный launcher для Xeon E5-2640 v4 (30 ядер максимум)
Специально настроенный для стабильной работы с ограничением ресурсов
"""

import os
import sys
import subprocess
import argparse
import psutil
import time
import shutil


def check_system_requirements():
    """Проверка системных требований"""

    print("🔍 Проверка системы...")

    # Проверка CPU
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()

    print(f"   CPU: {cpu_count} ядер")
    if cpu_freq:
        print(f"   Частота: {cpu_freq.current:.0f} MHz (макс: {cpu_freq.max:.0f} MHz)")

    if cpu_count < 20:
        print("⚠️ Рекомендуется минимум 20 ядер для эффективного обучения")

    # Проверка памяти
    memory = psutil.virtual_memory()
    memory_gb = memory.total / 1024 ** 3
    print(f"   RAM: {memory_gb:.1f} GB (доступно: {memory.available / 1024 ** 3:.1f} GB)")

    if memory_gb < 16:
        print("⚠️ Рекомендуется минимум 16 GB RAM")
        return False

    # Проверка свободного места
    disk = psutil.disk_usage('.')
    free_gb = disk.free / 1024 ** 3
    print(f"   Диск: {free_gb:.1f} GB свободно")

    if free_gb < 10:
        print("⚠️ Недостаточно места на диске (нужно минимум 10 GB)")
        return False

    print("✅ Системные требования выполнены")
    return True


def setup_xeon_environment(max_cores=30):
    """Настройка окружения специально для Xeon E5-2640 v4"""

    print(f"🔧 Настройка окружения для {max_cores} ядер...")

    # Базовые настройки CPU
    os.environ['OMP_NUM_THREADS'] = str(max_cores)
    os.environ['MKL_NUM_THREADS'] = str(max_cores)
    os.environ['NUMEXPR_NUM_THREADS'] = str(max_cores)

    # Intel MKL оптимизации для Xeon
    os.environ['MKL_DYNAMIC'] = 'FALSE'
    os.environ['MKL_DOMAIN_NUM_THREADS'] = f"MKL_BLAS={max_cores // 2},MKL_FFT={max_cores // 4}"

    # CPU affinity для NUMA
    # E5-2640 v4 имеет 2 NUMA узла
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    os.environ['KMP_BLOCKTIME'] = '1'

    # Отключаем некоторые оптимизации которые могут мешать
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    print(f"   OMP_NUM_THREADS: {max_cores}")
    print(f"   MKL_NUM_THREADS: {max_cores}")
    print(f"   KMP_AFFINITY: настроен для NUMA")


def get_optimal_params_for_dataset_size(estimated_swaths):
    """Определение оптимальных параметров в зависимости от размера данных"""

    if estimated_swaths < 1000:
        # Маленький датасет
        return {
            'batch_size': 8,
            'num_workers': 12,
            'epochs': 30,
            'description': 'Маленький датасет (быстрое обучение)'
        }
    elif estimated_swaths < 5000:
        # Средний датасет
        return {
            'batch_size': 6,
            'num_workers': 15,
            'epochs': 40,
            'description': 'Средний датасет (оптимальный баланс)'
        }
    elif estimated_swaths < 15000:
        # Большой датасет
        return {
            'batch_size': 6,
            'num_workers': 15,
            'epochs': 50,
            'description': 'Большой датасет (высокое качество)'
        }
    else:
        # Очень большой датасет
        return {
            'batch_size': 4,
            'num_workers': 12,
            'epochs': 50,
            'description': 'Очень большой датасет (максимальное качество)'
        }


def estimate_training_time(swaths, epochs, batch_size, cpu_cores=30):
    """Примерная оценка времени обучения"""

    # Базовая оценка: время на один swath в секундах
    base_time_per_swath = 0.8  # секунд на swath на одно ядро

    # Коррекция на количество ядер (не линейная из-за overhead)
    cpu_efficiency = min(1.0, cpu_cores / 40) * 0.85  # 85% эффективность

    # Коррекция на batch size
    batch_efficiency = min(1.0, batch_size / 8) * 0.9

    # Общее время в секундах
    total_seconds = (swaths * epochs * base_time_per_swath) / (cpu_efficiency * batch_efficiency)

    hours = total_seconds / 3600
    days = hours / 24

    return hours, days


def create_monitoring_script():
    """Создание скрипта для мониторинга обучения"""

    script_content = '''#!/bin/bash

echo "🖥️  Мониторинг AMSR2 обучения на Xeon E5-2640 v4"
echo "==============================================="

# Цвета для вывода
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
RED='\\033[0;31m'
NC='\\033[0m' # No Color

while true; do
    clear
    echo "🖥️  Мониторинг AMSR2 обучения на Xeon E5-2640 v4"
    echo "==============================================="
    echo "Время: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # CPU информация
    echo "📊 CPU Status:"
    cpu_usage=$(mpstat 1 1 | tail -1 | awk '{print 100-$12}')
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        color=$GREEN
    elif (( $(echo "$cpu_usage > 50" | bc -l) )); then
        color=$YELLOW
    else
        color=$RED
    fi
    printf "   Общая загрузка: ${color}%.1f%%${NC}\\n" $cpu_usage

    # Проверка использования 30 ядер
    busy_cores=$(mpstat -P ALL 1 1 | grep "Average" | awk '$12 < 50' | wc -l)
    printf "   Активных ядер: %d из 40 (цель: ~30)\\n" $busy_cores

    # Memory
    echo ""
    echo "💾 Memory Status:"
    memory_info=$(free | grep "Mem:")
    total_mem=$(echo $memory_info | awk '{print $2}')
    used_mem=$(echo $memory_info | awk '{print $3}')
    mem_percent=$(echo "scale=1; $used_mem * 100 / $total_mem" | bc)

    if (( $(echo "$mem_percent > 90" | bc -l) )); then
        color=$RED
    elif (( $(echo "$mem_percent > 70" | bc -l) )); then
        color=$YELLOW  
    else
        color=$GREEN
    fi
    printf "   RAM использование: ${color}%.1f%%${NC}\\n" $mem_percent

    # PyTorch процессы
    echo ""
    echo "🐍 Python/PyTorch процессы:"
    python_procs=$(ps aux | grep python | grep -v grep | grep -v monitor | wc -l)
    echo "   Активных Python процессов: $python_procs"

    # Топ процессы по CPU
    echo "   Топ CPU процессы:"
    ps aux --sort=-%cpu | head -4 | tail -3 | while read line; do
        echo "     $line"
    done

    # Training progress
    echo ""
    echo "🎯 Training Progress:"
    if [ -f "training.log" ]; then
        tail -3 training.log | grep -E "(Эпоха|PSNR|Loss)" | tail -2
    elif [ -f "nohup.out" ]; then
        tail -3 nohup.out | grep -E "(Эпоха|PSNR|Loss)" | tail -2
    else
        echo "   Логи обучения не найдены"
    fi

    # Файлы результатов
    echo ""
    echo "📁 Файлы результатов:"
    if [ -f "best_amsr2_model.pth" ]; then
        model_size=$(ls -lh best_amsr2_model.pth | awk '{print $5}')
        echo "   ✅ Модель: best_amsr2_model.pth ($model_size)"
    else
        echo "   ⏳ Модель еще не сохранена"
    fi

    if [ -f "amsr2_training_history.png" ]; then
        echo "   ✅ График: amsr2_training_history.png"
    fi

    # Системная нагрузка
    echo ""
    echo "⚡ Системная нагрузка:"
    uptime | awk '{print "   Load average: " $(NF-2) " " $(NF-1) " " $NF}'

    echo ""
    echo "Обновление каждые 30 секунд. Для выхода: Ctrl+C"
    echo "==============================================="

    sleep 30
done
'''

    with open('monitor_training.sh', 'w') as f:
        f.write(script_content)

    os.chmod('monitor_training.sh', 0o755)
    print("📊 Создан скрипт мониторинга: monitor_training.sh")


def main():
    """Главная функция launcher"""

    parser = argparse.ArgumentParser(
        description='Launcher для AMSR2 Super-Resolution на Xeon E5-2640 v4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:

1. Быстрый тест (30-45 минут):
   python run_amsr2_xeon_optimized.py --quick-test --npz-dir /path/to/data

2. Средний тест на реальных данных (4-8 часов):
   python run_amsr2_xeon_optimized.py --medium-test --npz-dir /path/to/data

3. Полное обучение (48-84 часа):
   python run_amsr2_xeon_optimized.py --full-train --npz-dir /path/to/data

4. Кастомные параметры:
   python run_amsr2_xeon_optimized.py --full-train --npz-dir /path/to/data \\
       --batch-size 4 --epochs 40 --max-cores 25
        '''
    )

    # Режимы работы
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--quick-test', action='store_true',
                       help='Быстрый тест с синтетическими данными (30-45 мин)')
    group.add_argument('--medium-test', action='store_true',
                       help='Средний тест на реальных данных (4-8 часов)')
    group.add_argument('--full-train', action='store_true',
                       help='Полное обучение на всех данных (2-4 дня)')

    # Основные параметры
    parser.add_argument('--npz-dir', type=str,
                        help='Путь к NPZ файлам (обязательно для medium-test и full-train)')

    # Настройки производительности
    parser.add_argument('--max-cores', type=int, default=30,
                        help='Максимум CPU ядер (по умолчанию 30)')
    parser.add_argument('--batch-size', type=int,
                        help='Размер batch (автоматически оптимизируется)')
    parser.add_argument('--epochs', type=int,
                        help='Количество эпох (автоматически оптимизируется)')
    parser.add_argument('--workers', type=int,
                        help='DataLoader workers (автоматически оптимизируется)')

    # Дополнительные параметры
    parser.add_argument('--orbit-filter', choices=['A', 'D', 'U'],
                        help='Фильтр по типу орбиты')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--scale-factor', type=int, default=10,
                        help='Масштаб увеличения')

    # Флаги
    parser.add_argument('--no-monitoring', action='store_true',
                        help='Не создавать скрипт мониторинга')
    parser.add_argument('--background', action='store_true',
                        help='Запуск в фоновом режиме с nohup')

    args = parser.parse_args()

    print("🛰️ AMSR2 SUPER-RESOLUTION - XEON E5-2640 V4 OPTIMIZED")
    print("=" * 60)

    # Проверка системы
    if not check_system_requirements():
        print("❌ Системные требования не выполнены")
        sys.exit(1)

    # Валидация аргументов
    if (args.medium_test or args.full_train) and not args.npz_dir:
        print("❌ Для medium-test и full-train требуется --npz-dir")
        sys.exit(1)

    if args.npz_dir and not os.path.exists(args.npz_dir):
        print(f"❌ Директория не найдена: {args.npz_dir}")
        sys.exit(1)

    # Ограничение CPU
    max_cores = min(args.max_cores, 30)
    if max_cores != args.max_cores:
        print(f"⚠️ Ограничиваем CPU до {max_cores} ядер для стабильности")

    # Настройка окружения
    setup_xeon_environment(max_cores)

    # Оценка размера датасета для оптимизации параметров
    if args.npz_dir:
        npz_files = len([f for f in os.listdir(args.npz_dir) if f.endswith('.npz')])
        estimated_swaths = npz_files * 10  # Примерно 10 swaths на файл
        print(f"📊 Найдено {npz_files} NPZ файлов (~{estimated_swaths} swaths)")
    else:
        estimated_swaths = 100  # Для тестового режима

    # Получение оптимальных параметров
    optimal = get_optimal_params_for_dataset_size(estimated_swaths)
    print(f"⚙️  Режим: {optimal['description']}")

    # Определение финальных параметров
    final_batch_size = args.batch_size or optimal['batch_size']
    final_workers = args.workers or optimal['num_workers']

    # Построение команды
    cmd = [sys.executable, "amsr2_super_resolution_complete.py"]

    if args.quick_test:
        print(f"\n🚀 РЕЖИМ: Быстрый тест ({max_cores} ядер)")
        cmd.extend([
            "--test-mode",
            "--epochs", "10",
            "--batch-size", str(final_batch_size),
            "--cpu-workers", str(final_workers),
            "--cpu-threads", str(max_cores),
            "--max-cpu-cores", str(max_cores)
        ])

        # Оценка времени
        hours, days = estimate_training_time(100, 10, final_batch_size, max_cores)
        print(f"⏱️  Примерное время: {hours:.1f} часов")

    elif args.medium_test:
        final_epochs = args.epochs or 25
        print(f"\n🧪 РЕЖИМ: Средний тест ({max_cores} ядер)")
        cmd.extend([
            "--npz-dir", args.npz_dir,
            "--epochs", str(final_epochs),
            "--batch-size", str(final_batch_size),
            "--cpu-workers", str(final_workers),
            "--cpu-threads", str(max_cores),
            "--max-cpu-cores", str(max_cores)
        ])

        # Оценка времени
        hours, days = estimate_training_time(estimated_swaths, final_epochs, final_batch_size, max_cores)
        print(f"⏱️  Примерное время: {hours:.1f} часов ({days:.1f} дней)")

    elif args.full_train:
        final_epochs = args.epochs or optimal['epochs']
        print(f"\n🎯 РЕЖИМ: Полное обучение ({max_cores} ядер)")
        cmd.extend([
            "--npz-dir", args.npz_dir,
            "--epochs", str(final_epochs),
            "--batch-size", str(final_batch_size),
            "--cpu-workers", str(final_workers),
            "--cpu-threads", str(max_cores),
            "--max-cpu-cores", str(max_cores)
        ])

        # Оценка времени
        hours, days = estimate_training_time(estimated_swaths, final_epochs, final_batch_size, max_cores)
        print(f"⏱️  Примерное время: {hours:.1f} часов ({days:.1f} дней)")

        if days > 5:
            print("⚠️ Обучение займет очень много времени. Рассмотрите:")
            print("   - Уменьшение --epochs")
            print("   - Увеличение --batch-size")
            print("   - Использование --medium-test для начала")

    # Добавление дополнительных параметров
    if args.orbit_filter:
        cmd.extend(["--orbit-filter", args.orbit_filter])

    if args.lr != 1e-4:
        cmd.extend(["--lr", str(args.lr)])

    if args.scale_factor != 10:
        cmd.extend(["--scale-factor", str(args.scale_factor)])

    # Вывод конфигурации
    print(f"\n⚙️  Конфигурация:")
    print(f"   CPU ядра: {max_cores} из {psutil.cpu_count()}")
    print(f"   DataLoader workers: {final_workers}")
    print(f"   Batch size: {final_batch_size}")
    print(f"   Эпохи: {cmd[cmd.index('--epochs') + 1] if '--epochs' in cmd else 'auto'}")

    # Создание мониторинга
    if not args.no_monitoring:
        create_monitoring_script()
        print(f"\n📊 Для мониторинга запустите в отдельном терминале:")
        print("   ./monitor_training.sh")

    # Финальная команда
    print(f"\n🚀 Команда запуска:")
    print(" ".join(cmd))

    # Запуск
    print("\n" + "=" * 60)

    try:
        if args.background:
            print("🔄 Запуск в фоновом режиме...")
            with open('training.log', 'w') as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                )
            print(f"✅ Процесс запущен (PID: {process.pid})")
            print("📝 Логи: tail -f training.log")
            print("📊 Мониторинг: ./monitor_training.sh")
        else:
            result = subprocess.run(cmd, check=True)
            print("\n" + "=" * 60)
            print("🎉 ВЫПОЛНЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ ОШИБКА ВЫПОЛНЕНИЯ (код: {e.returncode})")
        print("Проверьте логи для диагностики")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Выполнение прервано пользователем")
        sys.exit(1)


if __name__ == "__main__":
    main()