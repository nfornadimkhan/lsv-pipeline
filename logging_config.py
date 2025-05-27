"""
Logging configuration module for the PDF processing pipeline.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create a timestamp for the log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"pdf_pipeline_{timestamp}.log"

# Define custom theme for Rich console
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "debug": "green",
    "success": "green",
})

# Create Rich console with custom theme
console = Console(theme=custom_theme)

def setup_logging():
    """Configure logging with both file and console handlers."""
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    
    # Rich console handler with custom formatting
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        rich_tracebacks=True,
        markup=True
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    root_logger.handlers = []
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create logger for the PDF pipeline
    logger = logging.getLogger("pdf_pipeline")
    
    # Log the start of a new session
    logger.info(f"Starting new logging session. Log file: {log_file}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name) 