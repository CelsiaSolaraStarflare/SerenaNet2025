"""
Logging utilities for SerenaNet.

This module provides centralized logging configuration and utilities
for consistent logging across the entire project.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import datetime


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file (str, optional): Path to log file
        log_dir (str, optional): Directory for log files
        format_string (str, optional): Custom format string
        include_timestamp (bool): Whether to include timestamp in log filename
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create log directory if specified
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        if not log_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir_path / f"serenanet_{timestamp}.log"
    
    # Default format string
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    # Create formatters
    formatter = logging.Formatter(format_string)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file}")
    
    # Set specific logger levels for external libraries
    external_loggers = {
        'transformers': logging.WARNING,
        'torch': logging.WARNING,
        'torchaudio': logging.WARNING,
        'sklearn': logging.WARNING,
        'matplotlib': logging.WARNING,
        'PIL': logging.WARNING
    }
    
    for logger_name, logger_level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(logger_level)
    
    logging.info("Logging configured successfully")
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name (str): Logger name (usually __name__)
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for console output.
    """
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.RESET}"
            )
        
        return super().format(record)


def setup_colored_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging with colored console output.
    
    Args:
        level (int): Logging level
        log_file (str, optional): Path to log file
        
    Returns:
        logging.Logger: Configured logger
    """
    # Format strings
    console_format = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)d - %(message)s"
    )
    file_format = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)d - %(message)s"
    )
    
    # Create formatters
    colored_formatter = ColoredFormatter(console_format)
    file_formatter = logging.Formatter(file_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(colored_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


class TrainingLogger:
    """
    Specialized logger for training metrics and progress.
    
    Args:
        name (str): Logger name
        log_file (str, optional): Path to log file
        tensorboard_dir (str, optional): TensorBoard log directory
    """
    
    def __init__(
        self,
        name: str = "training",
        log_file: Optional[str] = None,
        tensorboard_dir: Optional[str] = None
    ):
        self.logger = get_logger(name)
        self.log_file = log_file
        
        # TensorBoard writer (optional)
        self.tb_writer = None
        if tensorboard_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
            except ImportError:
                self.logger.warning("TensorBoard not available")
        
        # Training state
        self.step = 0
        self.epoch = 0
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log training metrics.
        
        Args:
            metrics (Dict[str, float]): Metrics to log
            step (int, optional): Training step
            prefix (str): Metric name prefix
        """
        if step is not None:
            self.step = step
        
        # Log to console/file
        metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {self.step} - {metric_str}")
        
        # Log to TensorBoard
        if self.tb_writer:
            for name, value in metrics.items():
                metric_name = f"{prefix}/{name}" if prefix else name
                self.tb_writer.add_scalar(metric_name, value, self.step)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.epoch = epoch
        self.logger.info(f"Starting epoch {epoch}/{total_epochs}")
    
    def log_epoch_end(self, metrics: Dict[str, float]):
        """Log epoch end with metrics."""
        metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {self.epoch} completed - {metric_str}")
    
    def log_model_info(self, model):
        """Log model information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model: {model.__class__.__name__}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def close(self):
        """Close TensorBoard writer."""
        if self.tb_writer:
            self.tb_writer.close()


def log_system_info():
    """Log system information."""
    logger = get_logger("system")
    
    # Python info
    import sys
    logger.info(f"Python version: {sys.version}")
    
    # PyTorch info
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"GPU {i}: {gpu_name}")
    except ImportError:
        logger.warning("PyTorch not available")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"System memory: {memory.total / (1024**3):.1f} GB")
        logger.info(f"Available memory: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        logger.warning("psutil not available for memory info")


def create_experiment_logger(
    experiment_name: str,
    log_dir: str,
    level: int = logging.INFO
) -> TrainingLogger:
    """
    Create a logger for a specific experiment.
    
    Args:
        experiment_name (str): Name of the experiment
        log_dir (str): Directory for log files
        level (int): Logging level
        
    Returns:
        TrainingLogger: Configured training logger
    """
    # Create experiment log directory
    exp_log_dir = Path(log_dir) / experiment_name
    exp_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup main logging
    log_file = exp_log_dir / "training.log"
    setup_logging(level=level, log_file=str(log_file))
    
    # Create training logger
    tensorboard_dir = exp_log_dir / "tensorboard"
    training_logger = TrainingLogger(
        name=f"training.{experiment_name}",
        log_file=str(log_file),
        tensorboard_dir=str(tensorboard_dir)
    )
    
    # Log system info
    log_system_info()
    
    return training_logger
