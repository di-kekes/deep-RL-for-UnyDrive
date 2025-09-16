import logging
import os


def setup_logger(log_dir="ep_logs", filename="train.log"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)

    # Вывод в файл
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setLevel(logging.INFO)

    # Вывод в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Формат
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Добавляем обработчики
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

