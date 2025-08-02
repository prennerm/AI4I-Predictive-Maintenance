"""
Logging utilities for AI4I Predictive Maintenance Project

Provides centralized logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None, 
                level: str = "INFO", console: bool = True) -> logging.Logger:
    """
    Set up a logger with specified configuration.
    
    Args:
        name (str): Logger name
        log_file (str, optional): Path to log file
        level (str): Logging level
        console (bool): Whether to log to console
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_project_logger(module_name: str) -> logging.Logger:
    """
    Get a standardized logger for project modules.
    
    Args:
        module_name (str): Name of the module
        
    Returns:
        logging.Logger: Configured project logger
    """
    return setup_logger(f"ai4i.{module_name}", level="INFO")
