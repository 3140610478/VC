import logging
import os
import sys
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
base_folder_name = os.path.split(base_folder)[-1]
if base_folder not in sys.path:
    sys.path.append(base_folder)
log_path = os.path.abspath(os.path.join(base_folder, "./Log"))
if not os.path.exists(log_path):
    os.mkdir(log_path)


def getLogger(name: str, mode="w") -> logging.Logger:
    logger = logging.getLogger(f"[{name}] logger in {base_folder_name}")
    logger.setLevel(logging.INFO)
    terminal_handler = logging.StreamHandler(sys.stderr)
    terminal_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(
        os.path.abspath(os.path.join(log_path, f"./{name}.log")),
        mode=mode,
    )
    file_handler.setLevel(logging.INFO)
    logger.addHandler(terminal_handler)
    logger.addHandler(file_handler)

    return logger
