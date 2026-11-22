import logging
import os


def get_root_logger(logger_name="basicsr", log_level=logging.INFO, log_file=None):
    return logging.getLogger(logger_name)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """BasicSR's scandir: yields file paths under ``dir_path`` (relative unless ``full_path``)."""
    for entry in os.scandir(dir_path):
        if entry.is_file() and not entry.name.startswith("."):
            path = entry.path if full_path else os.path.relpath(entry.path, dir_path)
            if suffix is None or path.endswith(suffix):
                yield path
        elif recursive and entry.is_dir():
            yield from scandir(entry.path, suffix, recursive, full_path)
