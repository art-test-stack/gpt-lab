import logging, sys, os
import re
from typing import Union, Tuple
from .common import format_value, get_rank, is_rank0


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

logger = logging.getLogger()

_LOG_LEVELS: Tuple[int, int, int, int, int] = tuple(v for v in log_levels.values())

def _get_level(level: Union[str, int]) -> int:
    if level not in _LOG_LEVELS and not isinstance(level, str):
        raise ValueError(f"Invalid log level: {level}. Must be one of {list(log_levels.keys())} or an integer log level.")
    if isinstance(level, str):
        return log_levels.get(level.upper(), logging.INFO)
    return level

# adapted from torchtitan logging utility
# https://github.com/pytorch/torchtitan/blob/0943771/torchtitan/tools/logging.py#L15
def init_logger() -> None:
    log_level = os.getenv("GPTLAB_LOG_LEVEL", "INFO").upper()
    print("Environment variable GPTLAB_LOG_LEVEL:", os.getenv("GPTLAB_LOG_LEVEL"))
    print("GPTLAB_LOG_LEVEL:", log_level)
    log_level = getattr(logging, log_level, logging.INFO)
    print(f"log_level: {log_level}")
    logger.setLevel(log_level)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    formatter = ColoredFormatter(
        "[gpt-lab] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"

_logged: set[str] = set() # global set to track logged messages for warn_once

def warn_once(logger: logging.Logger, msg: str) -> None:
    """Log a warning message only once per unique message.

    Uses a global set to track messages that have already been logged
    to prevent duplicate warning messages from cluttering the output.

    Args:
        logger (logging.Logger): The logger instance to use for warning.
        msg (str): The warning message to log.
    """
    if msg not in _logged:
        logger.warning(msg)
        _logged.add(msg)


def debug0(message, logger=logger):
    if is_rank0():
        logger.debug(message, stacklevel=3)

def log0(message, logger=logger, level=logging.INFO, stacklevel=3):
    level = _get_level(level)
    if is_rank0():
        logger.log(level, message, stacklevel=stacklevel)

def _with_rank(msg):
    rank = get_rank()
    return f"[RANK {rank}] {msg}" if rank != 0 else msg

def error(message, logger=logger):
    logger.error(_with_rank(message), stacklevel=3)

def log_error(message, error_type=ValueError, logger=logger):
    logger.error(_with_rank(message), stacklevel=3)
    raise error_type(message)

def log_critical(message, error_type=RuntimeError, logger=logger):
    logger.critical(_with_rank(message), stacklevel=3)
    raise error_type(message)

def log_all(msg, level=logging.ERROR, logger=logger):
    if isinstance(level, str):
        level = log_levels.get(level.upper(), logging.ERROR)
    logger.log(level, _with_rank(msg), stacklevel=3)
    if level >= logging.ERROR:
        raise RuntimeError(msg)

def log_dict(title, info, logger=logger, level=logging.INFO, only_rank0=True, structured=False):
    level = _get_level(level)
    if only_rank0 and not is_rank0():
        return

    if structured:
        logger.log(level, f"{title} | {info}")
        return

    lines = [f"{title}:"]
    for k, v in info.items():
        lines.append(f"\t{k:<25}: {format_value(v):<60}")

    msg = "\n".join(lines)
    
    if not only_rank0:
        msg = _with_rank(msg)

    logger.log(level, msg, stacklevel=4)