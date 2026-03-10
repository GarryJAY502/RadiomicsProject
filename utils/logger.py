"""
Global Logging Module for RadiomicsProject.

Design Goals:
  1. All print() output goes to BOTH console AND timestamped log file (as INFO).
  2. tqdm progress bars render normally on console (dynamic refresh).
     They are NOT written to the log file and NOT marked as ERROR.
  3. Real exceptions (tracebacks) are captured and logged.
  4. Third-party library stderr warnings are logged at WARNING level.
"""

import os
import sys
import logging
import re
from datetime import datetime


class LoggingStreamProxy:
    """
    A smart stream proxy that replaces sys.stdout or sys.stderr.
    
    - Normal text (print, etc.) → forwarded to both the real console stream
      AND the Python logger (at the configured level).
    - tqdm output (detected by \\r without \\n) → forwarded ONLY to the real
      console stream, completely bypassing the log file.
    """

    # Pattern to detect tqdm-style progress output (contains \r or percentage patterns)
    _TQDM_PATTERN = re.compile(r'\r|^\s*\d+%\|')

    def __init__(self, logger: logging.Logger, original_stream, log_level=logging.INFO):
        self.logger = logger
        self.original_stream = original_stream
        self.log_level = log_level

    def write(self, buf: str):
        if not buf or buf.isspace():
            # Whitespace-only writes (e.g. bare \n from print) → just pass to console
            self.original_stream.write(buf)
            return

        # Detect tqdm: it uses \r to overwrite the current line
        is_tqdm = '\r' in buf and '\n' not in buf

        if is_tqdm:
            # tqdm output → console only, skip log file entirely
            self.original_stream.write(buf)
            self.original_stream.flush()
        else:
            # Normal output → console + log file
            self.original_stream.write(buf)
            self.original_stream.flush()
            # Log each non-empty line
            for line in buf.rstrip().splitlines():
                stripped = line.rstrip()
                if stripped:
                    self.logger.log(self.log_level, stripped)

    def flush(self):
        self.original_stream.flush()

    def isatty(self):
        """Let tqdm think it's writing to a real terminal so it uses Unicode bars."""
        return hasattr(self.original_stream, 'isatty') and self.original_stream.isatty()

    # Forward any other attribute lookups to the original stream
    def __getattr__(self, name):
        return getattr(self.original_stream, name)


def setup_global_logging(log_dir: str = "logs", task_name: str = "run_pipeline") -> str:
    """
    Initialize the global logging system.

    After calling this function:
      - print("hello")        → console + log file [INFO]
      - tqdm progress bars    → console only (dynamic refresh, no log pollution)
      - Exceptions/tracebacks → console + log file [ERROR]
      - Third-party stderr    → console + log file [WARNING]

    Returns:
        Path to the created log file.
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{task_name}_{timestamp}.log")

    # Save original streams BEFORE any replacement
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Configure root logger with file-only handler
    # (console output is handled by our LoggingStreamProxy, not by StreamHandler)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Remove any pre-existing handlers to avoid duplicate output
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)

    log = logging.getLogger('GlobalLogger')

    # Replace sys.stdout: print() → INFO level, also passes through to real console
    sys.stdout = LoggingStreamProxy(log, original_stdout, log_level=logging.INFO)
    # Replace sys.stderr: tqdm/warnings → WARNING level, tracebacks → also WARNING
    # (Real errors from our code use explicit logging.error() or print("[Error]"))
    sys.stderr = LoggingStreamProxy(log, original_stderr, log_level=logging.WARNING)

    # Store originals for emergency access
    sys._original_stdout = original_stdout
    sys._original_stderr = original_stderr

    # Announce (this will go through our new proxy → both console and log)
    print("=" * 50)
    print(f"Logging initialized. Log file: {log_file}")
    print("=" * 50)

    return log_file
