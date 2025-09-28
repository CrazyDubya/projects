"""
Tests for shared utilities module.

This module tests the common functionality provided by shared_utils.py
to ensure reliability across all projects.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the shared utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared_utils import (
    ProjectConfig,
    setup_logging,
    RateLimiter,
    safe_file_operation,
    validate_api_key,
    timestamp_filename,
    ensure_directory
)


class TestProjectConfig:
    """Test configuration management functionality."""
    
    def test_config_initialization(self):
        """Test that configuration initializes properly."""
        config = ProjectConfig()
        assert config is not None
        assert isinstance(config.config, dict)
    
    def test_config_with_file(self):
        """Test configuration loading from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_config = {"test_key": "test_value", "number": 42}
            json.dump(test_config, f)
            config_file = f.name
        
        try:
            config = ProjectConfig(config_file)
            assert config.get("test_key") == "test_value"
            assert config.get("number") == 42
        finally:
            os.unlink(config_file)
    
    def test_config_env_override(self):
        """Test that environment variables override config file."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG', 'TEST_VAR': 'env_value'}):
            config = ProjectConfig()
            assert config.get('log_level') == 'DEBUG'
    
    def test_config_get_default(self):
        """Test getting config values with defaults."""
        config = ProjectConfig()
        assert config.get('nonexistent_key', 'default') == 'default'
        assert config.get('nonexistent_key') is None


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_allows_requests(self):
        """Test that rate limiter allows requests within limits."""
        limiter = RateLimiter(max_requests=5, time_window=10)
        
        # Should allow first 5 requests
        for _ in range(5):
            assert limiter.can_make_request()
            limiter.make_request()
        
        # 6th request should be denied
        assert not limiter.can_make_request()
    
    def test_rate_limiter_resets_after_time_window(self):
        """Test that rate limiter resets after time window."""
        limiter = RateLimiter(max_requests=2, time_window=1)
        
        # Make 2 requests
        limiter.make_request()
        limiter.make_request()
        
        # Should be at limit
        assert not limiter.can_make_request()
        
        # Wait for time window to pass
        import time
        time.sleep(1.1)
        
        # Should be able to make requests again
        assert limiter.can_make_request()


class TestSafeFileOperations:
    """Test safe file operation functionality."""
    
    def test_file_exists_operation(self):
        """Test file existence checking."""
        # Test with non-existent file
        result = safe_file_operation('exists', 'nonexistent_file.txt')
        assert result is False
        
        # Test with existing file
        with tempfile.NamedTemporaryFile() as temp_file:
            result = safe_file_operation('exists', temp_file.name)
            assert result is True
    
    def test_file_write_and_read_operations(self):
        """Test file writing and reading."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Test writing
            test_content = "Hello, World!"
            result = safe_file_operation('write', temp_path, test_content)
            assert result is True
            
            # Test reading
            content = safe_file_operation('read', temp_path)
            assert content == test_content
        finally:
            os.unlink(temp_path)
    
    def test_json_file_operations(self):
        """Test JSON file operations."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Test writing JSON
            test_data = {"key": "value", "number": 42}
            result = safe_file_operation('write', temp_path, test_data)
            assert result is True
            
            # Test reading JSON
            data = safe_file_operation('read', temp_path)
            assert data == test_data
        finally:
            os.unlink(temp_path)
    
    def test_file_operation_error_handling(self):
        """Test error handling in file operations."""
        # Test invalid operation
        result = safe_file_operation('invalid_op', 'test.txt')
        assert result is None
        
        # Test reading non-existent file
        result = safe_file_operation('read', 'nonexistent.txt')
        assert result is None


class TestAPIKeyValidation:
    """Test API key validation functionality."""
    
    def test_valid_api_key(self):
        """Test validation of valid API key."""
        valid_key = "sk-1234567890abcdef"
        assert validate_api_key(valid_key, "Test Service") is True
    
    def test_invalid_api_key(self):
        """Test validation of invalid API keys."""
        # Test None key
        assert validate_api_key(None, "Test Service") is False
        
        # Test empty key
        assert validate_api_key("", "Test Service") is False
        
        # Test too short key
        assert validate_api_key("short", "Test Service") is False


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_timestamp_filename(self):
        """Test timestamp filename generation."""
        filename = timestamp_filename("test", ".txt")
        assert filename.startswith("test_")
        assert filename.endswith(".txt")
        assert len(filename) > 10  # Should include timestamp
    
    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "new_dir" / "sub_dir"
            
            # Directory shouldn't exist initially
            assert not test_path.exists()
            
            # Create directory
            result = ensure_directory(test_path)
            
            # Directory should now exist
            assert test_path.exists()
            assert test_path.is_dir()
            assert result == test_path


class TestLoggingSetup:
    """Test logging setup functionality."""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        logger = setup_logging()
        assert logger is not None
        assert logger.level <= 20  # INFO level or below
    
    def test_setup_logging_with_level(self):
        """Test logging setup with specific level."""
        logger = setup_logging(level="DEBUG")
        assert logger.level <= 10  # DEBUG level
        
        logger = setup_logging(level="ERROR")
        assert logger.level <= 40  # ERROR level
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            log_file = temp_file.name
        
        try:
            logger = setup_logging(log_file=log_file)
            
            # Test that log file is created
            logger.info("Test message")
            assert os.path.exists(log_file)
            
            # Check that message was written to file
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)


# Integration tests
class TestIntegration:
    """Test integration between different utilities."""
    
    def test_config_and_logging_integration(self):
        """Test that configuration works with logging setup."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG'}):
            config = ProjectConfig()
            log_level = config.get('log_level')
            logger = setup_logging(level=log_level)
            
            assert log_level == 'DEBUG'
            assert logger.level <= 10  # DEBUG level
    
    def test_file_operations_with_ensure_directory(self):
        """Test file operations with directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "new_subdir"
            test_file = test_dir / "test.txt"
            
            # Ensure directory exists
            ensure_directory(test_dir)
            
            # Write file
            result = safe_file_operation('write', test_file, "Test content")
            assert result is True
            
            # Read file
            content = safe_file_operation('read', test_file)
            assert content == "Test content"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])