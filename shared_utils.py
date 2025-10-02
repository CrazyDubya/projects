"""
Shared utilities for the projects repository.

This module provides common functionality used across multiple projects including:
- Configuration management
- Logging setup
- Error handling
- File operations
- API utilities
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime


class ProjectConfig:
    """Centralized configuration management for projects."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.json"
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file and environment variables."""
        # Load from file if exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Failed to load config file {self.config_file}: {e}")
        
        # Override with environment variables
        self.config.update({
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'data_dir': os.getenv('DATA_DIR', './data'),
            'output_dir': os.getenv('OUTPUT_DIR', './output'),
        })
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback to default."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except IOError as e:
            logging.error(f"Failed to save config file {self.config_file}: {e}")


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up standardized logging for projects.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        format_string: Custom format string for logs
    
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[]
    )
    
    logger = logging.getLogger(__name__.split('.')[0])
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_string))
            logger.addHandler(file_handler)
        except IOError as e:
            logger.warning(f"Failed to create log file {log_file}: {e}")
    
    return logger


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_requests: int, time_window: int):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def can_make_request(self) -> bool:
        """Check if a request can be made within rate limits."""
        now = time.time()
        # Remove requests outside the time window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        return len(self.requests) < self.max_requests
    
    def make_request(self) -> None:
        """Record a request being made."""
        self.requests.append(time.time())
    
    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        if not self.can_make_request():
            oldest_request = min(self.requests)
            wait_time = self.time_window - (time.time() - oldest_request)
            if wait_time > 0:
                logging.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)


def safe_file_operation(operation: str, file_path: Union[str, Path], 
                       data: Any = None, encoding: str = 'utf-8') -> Optional[Any]:
    """
    Safely perform file operations with error handling.
    
    Args:
        operation: Type of operation ('read', 'write', 'append', 'exists')
        file_path: Path to the file
        data: Data to write (for write/append operations)
        encoding: File encoding
    
    Returns:
        File contents for read operations, True/False for exists, None for write operations
    """
    file_path = Path(file_path)
    
    try:
        if operation == 'exists':
            return file_path.exists()
        
        elif operation == 'read':
            if not file_path.exists():
                logging.warning(f"File does not exist: {file_path}")
                return None
            
            with open(file_path, 'r', encoding=encoding) as f:
                if file_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    return f.read()
        
        elif operation in ['write', 'append']:
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            mode = 'a' if operation == 'append' else 'w'
            with open(file_path, mode, encoding=encoding) as f:
                if isinstance(data, dict) and file_path.suffix.lower() == '.json':
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    f.write(str(data))
            
            return True
        
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    except Exception as e:
        logging.error(f"File operation '{operation}' failed for {file_path}: {e}")
        return None


def validate_api_key(api_key: Optional[str], service_name: str) -> bool:
    """
    Validate API key format and warn if missing.
    
    Args:
        api_key: The API key to validate
        service_name: Name of the service for error messages
    
    Returns:
        True if API key is valid, False otherwise
    """
    if not api_key:
        logging.error(f"{service_name} API key not found. Please set the appropriate environment variable.")
        return False
    
    if len(api_key.strip()) < 10:  # Basic length check
        logging.error(f"{service_name} API key appears to be invalid (too short).")
        return False
    
    return True


def timestamp_filename(base_name: str, extension: str = ".txt") -> str:
    """
    Generate a timestamped filename.
    
    Args:
        base_name: Base name for the file
        extension: File extension
    
    Returns:
        Timestamped filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{extension}"


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


# Global configuration instance
config = ProjectConfig()
logger = setup_logging(level=config.get('log_level', 'INFO'))


if __name__ == "__main__":
    # Example usage
    logger.info("Shared utilities module loaded")
    
    # Test configuration
    print("Configuration test:")
    print(f"Data directory: {config.get('data_dir')}")
    print(f"Log level: {config.get('log_level')}")
    
    # Test file operations
    test_file = Path("test_output.txt")
    safe_file_operation('write', test_file, "Test content")
    content = safe_file_operation('read', test_file)
    print(f"File test - wrote and read: {content}")
    
    # Cleanup
    if test_file.exists():
        test_file.unlink()