# Project Standards and Templates

This document defines standards and provides templates for creating new projects in this repository.

## Project Structure Template

When creating a new project, use this standard structure:

```
project_name/
├── README.md                    # Project documentation
├── main.py                      # Primary entry point
├── config/                      # Configuration files
│   ├── default.json            # Default configuration
│   └── example.env             # Environment variables example
├── src/                         # Source code (optional, for larger projects)
│   ├── __init__.py
│   ├── core.py                 # Main functionality
│   └── utils.py                # Project-specific utilities
├── tests/                       # Test files
│   ├── __init__.py
│   ├── test_main.py
│   └── test_utils.py
├── docs/                        # Additional documentation
│   ├── usage.md                # Usage examples
│   └── api.md                  # API documentation (if applicable)
├── data/                        # Sample or test data
│   └── example_input.txt
└── requirements.txt             # Project-specific dependencies (if needed)
```

## Code Standards

### 1. File Headers
All Python files should start with a descriptive header:

```python
#!/usr/bin/env python3
"""
Brief description of what this module does.

Longer description if needed, explaining the purpose, key features,
and any important implementation details.

Author: Your Name
Created: YYYY-MM-DD
Last Modified: YYYY-MM-DD
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory for shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_utils import setup_logging, ProjectConfig

# Module configuration
CONFIG = ProjectConfig()
LOGGER = setup_logging(level=CONFIG.get('log_level', 'INFO'))
```

### 2. Function Documentation
Use Google-style docstrings:

```python
def process_data(
    input_file: Path, 
    output_format: str = "json",
    validate: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Process data from input file and return formatted results.
    
    This function reads data from the specified input file, processes it
    according to the output format, and optionally validates the results.
    
    Args:
        input_file: Path to the input file to process
        output_format: Format for output data ('json', 'csv', 'txt')
        validate: Whether to validate the processed data
    
    Returns:
        Processed data dictionary, or None if processing failed
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If output_format is not supported
        ValidationError: If validation fails and validate=True
        
    Example:
        >>> result = process_data(Path("data.txt"), "json")
        >>> print(result["status"])
        "success"
    """
    pass
```

### 3. Error Handling Pattern
Use consistent error handling with logging:

```python
def safe_operation(param: str) -> bool:
    """Perform an operation with proper error handling."""
    try:
        # Main operation logic
        result = complex_operation(param)
        LOGGER.info(f"Operation completed successfully: {result}")
        return True
        
    except FileNotFoundError as e:
        LOGGER.error(f"Required file not found: {e}")
        return False
        
    except ValueError as e:
        LOGGER.error(f"Invalid parameter value: {e}")
        return False
        
    except Exception as e:
        LOGGER.error(f"Unexpected error in safe_operation: {e}")
        return False
```

### 4. Configuration Management
Use the shared configuration system:

```python
# Load configuration
CONFIG = ProjectConfig()

# Get configuration values with defaults
api_key = CONFIG.get('api_key')
max_retries = CONFIG.get('max_retries', 3)
output_dir = CONFIG.get('output_dir', './output')

# Validate required configuration
if not validate_api_key(api_key, 'ServiceName'):
    LOGGER.error("API key validation failed")
    sys.exit(1)
```

## README Template

Use this template for project README files:

```markdown
# Project Name

Brief description of what the project does and its main purpose.

## Features

- List key features
- Use bullet points
- Be specific about capabilities

## Installation

### Prerequisites
- Python 3.8+
- Additional system requirements if any

### Setup
```bash
# Clone repository (if standalone)
git clone https://github.com/CrazyDubya/projects.git
cd projects/project_name

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp config/example.env .env
# Edit .env file with your configuration
```

## Usage

### Basic Usage
```bash
python main.py --input data/example.txt --output results.json
```

### Advanced Options
```bash
python main.py --help
```

### Configuration
The project uses JSON configuration files in the `config/` directory:

```json
{
    "setting1": "value1",
    "setting2": 42,
    "setting3": ["list", "of", "values"]
}
```

Environment variables (optional):
- `API_KEY`: Your API key for external services
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## API Reference (if applicable)

### Main Functions

#### `main_function(param1, param2)`
Description of what the function does.

**Parameters:**
- `param1` (str): Description of parameter
- `param2` (int): Description of parameter

**Returns:**
- `dict`: Description of return value

**Example:**
```python
result = main_function("example", 42)
print(result)
```

## Examples

### Example 1: Basic Processing
```python
from project_name import main_function

result = main_function("input.txt")
print(f"Processed {result['count']} items")
```

### Example 2: Advanced Usage
```python
# More complex example here
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_main.py

# Run with coverage
python -m pytest --cov=. tests/
```

## Contributing

1. Follow the coding standards in `DEVELOPMENT_GUIDE.md`
2. Add tests for new functionality
3. Update documentation
4. Ensure all tests pass

## License

This project is part of the projects repository and is licensed under the MIT License.

## Changelog

### Version 1.0.0 (YYYY-MM-DD)
- Initial release
- Basic functionality implemented

### Version 1.1.0 (YYYY-MM-DD)
- Added feature X
- Fixed bug Y
- Improved performance
```

## Testing Standards

### 1. Test File Structure
```python
"""
Tests for project_name module.

This module contains comprehensive tests for all functionality
in the project_name package.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from project_name.main import main_function
from shared_utils import ProjectConfig


class TestMainFunctionality:
    """Test the main functionality of the project."""
    
    def test_basic_operation(self):
        """Test basic operation with valid input."""
        result = main_function("test_input")
        assert result is not None
        assert result['status'] == 'success'
    
    def test_error_handling(self):
        """Test error handling with invalid input."""
        result = main_function("")
        assert result is None
    
    @patch('project_name.main.external_api_call')
    def test_with_mocked_api(self, mock_api):
        """Test functionality with mocked external dependencies."""
        mock_api.return_value = {"data": "test"}
        result = main_function("test")
        assert result is not None
        mock_api.assert_called_once()


class TestConfiguration:
    """Test configuration management."""
    
    def test_config_loading(self):
        """Test that configuration loads properly."""
        config = ProjectConfig()
        assert config is not None
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ProjectConfig()
        assert config.get('nonexistent', 'default') == 'default'


# Integration tests
class TestIntegration:
    """Test integration with other components."""
    
    def test_file_operations(self):
        """Test file input/output operations."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            # Test file processing
            result = main_function(tmp.name)
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 2. Test Categories
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test performance characteristics

## Quality Checklist

Before submitting a new project or major changes, ensure:

### Code Quality
- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have proper docstrings
- [ ] Type hints are used where appropriate
- [ ] Error handling is comprehensive
- [ ] Logging is implemented properly

### Documentation
- [ ] README.md is complete and accurate
- [ ] Usage examples are provided
- [ ] API documentation exists (if applicable)
- [ ] Configuration options are documented

### Testing
- [ ] Unit tests cover main functionality
- [ ] Integration tests verify component interactions
- [ ] All tests pass
- [ ] Test coverage is reasonable (>80% for critical code)

### Configuration
- [ ] Uses shared_utils for common functionality
- [ ] Configuration is externalized
- [ ] Environment variables are documented
- [ ] Default values are sensible

### Security
- [ ] No hardcoded credentials
- [ ] Input validation is implemented
- [ ] File operations are safe
- [ ] API keys are properly managed

### Performance
- [ ] No obvious performance bottlenecks
- [ ] Large data is processed efficiently
- [ ] Memory usage is reasonable
- [ ] Rate limiting is implemented for API calls

## Integration Guidelines

### Using Shared Utilities
Always prefer shared utilities over reimplementing common functionality:

```python
# Good: Use shared utilities
from shared_utils import safe_file_operation, setup_logging, ProjectConfig

# Bad: Reimplement file operations
with open(filename, 'r') as f:
    content = f.read()
```

### Project Interoperability
Design projects to work well together:

```python
# Export data in standard formats
def export_results(data: Dict[str, Any], format: str = "json") -> Path:
    """Export results in a format that other projects can consume."""
    if format == "json":
        # Use shared utilities for consistent JSON handling
        return safe_file_operation('write', output_file, data)
```

### Common Patterns
Follow established patterns for:
- Configuration management
- Logging setup
- Error handling
- File operations
- API interactions

## Deployment Considerations

### Dependencies
- Minimize external dependencies
- Pin dependency versions in requirements.txt
- Document system-level dependencies

### Configuration
- Support multiple configuration sources
- Provide sensible defaults
- Document all configuration options

### Monitoring
- Include health checks for long-running processes
- Log important events and errors
- Support metrics collection where appropriate

---

*This document is regularly updated to reflect best practices and lessons learned from the repository.*