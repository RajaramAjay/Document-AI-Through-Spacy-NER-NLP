import os
import logging
from datetime import datetime
import toml

# Load configuration from TOML
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.toml'))
config = toml.load(config_path)
log_max_size_mb = config['logging']['max_size_mb']
logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))

logger_setup_done = False  # Global flag to track if logger has been set up


class CallFileHandler(logging.Handler):
    def __init__(self, base_filename, max_size_mb, logs_dir):
        super().__init__()
        self.base_filename = base_filename
        self.max_size_mb = max_size_mb
        self.logs_dir = logs_dir
        self.current_filename = None
        self.current_file = None
        self.init_log_file()

    def init_log_file(self):
        """Initialize or continue using the most recent log file."""
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        log_files = [
            os.path.join(self.logs_dir, f) for f in os.listdir(self.logs_dir) 
            if f.startswith(self.base_filename) and f.endswith(".log")
        ]
        log_files = sorted(log_files, key=os.path.getmtime, reverse=True)

        if log_files:
            latest_file = log_files[0]
            latest_file_size = os.path.getsize(latest_file) / (1024 * 1024)  # Convert to MB
            print(f"Checking file: {latest_file}, Size: {latest_file_size:.2f} MB")  # Debug statement

            if latest_file_size < self.max_size_mb:
                print(f"Continuing with existing log file: {latest_file}")  # Debug statement
                self.current_filename = latest_file
                self.current_file = open(self.current_filename, 'a')
                return

        self.start_new_file()

    def emit(self, record):
        """Write a log message to the current log file."""
        if self.current_file is None:
            self.start_new_file()

        msg = self.format(record)
        self.current_file.write(msg + '\n')
        self.current_file.flush()

    def start_new_file(self):
        """Start a new log file."""
        if self.current_file:
            self.current_file.close()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.current_filename = os.path.join(self.logs_dir, f"{self.base_filename}_{timestamp}.log")
        print(f"Creating new log file: {self.current_filename}")  # Debug statement
        self.current_file = open(self.current_filename, 'a')

    def close(self):
        """Close the current log file when the handler is closed."""
        if self.current_file:
            self.current_file.close()


def setup_logger():
    global logger_setup_done
    if logger_setup_done:
        return logging.getLogger('Image_Processing')

    logger = logging.getLogger('Image_Processing')
    logger.setLevel(logging.DEBUG)

    # Remove all existing handlers (if any)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create our custom file handler
    file_handler = CallFileHandler(base_filename="Log", max_size_mb=log_max_size_mb, logs_dir=logs_dir)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler (for output to the terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger_setup_done = True
    return logger
