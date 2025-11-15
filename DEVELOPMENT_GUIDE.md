# Development Setup Guide

This guide helps you set up a development environment for contributing to the projects repository.

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/CrazyDubya/projects.git
cd projects

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the repository root:
```bash
# API Keys (optional - only needed for AI projects)
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here

# Configuration
LOG_LEVEL=INFO
DATA_DIR=./data
OUTPUT_DIR=./output
```

### 3. Development Tools Setup
```bash
# Install development tools
pip install black flake8 mypy pytest pytest-cov

# Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Project Structure

### Repository Layout
```
projects/
├── README.md                    # Main repository documentation
├── INDEX.md                     # Project portfolio index
├── PROJECT_INDEX.md             # Detailed project analysis
├── TECHNICAL_EVALUATION.md      # Technical assessment and roadmap
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Tool configurations
├── shared_utils.py             # Common utilities library
├── .flake8                     # Linting configuration
├── project_name/               # Individual project directories
│   ├── README.md
│   ├── main.py
│   └── requirements.txt (if project-specific)
└── tests/                      # Test files
    └── test_project_name.py
```

### Individual Project Structure
Each project should follow this standard structure:
```
project_name/
├── README.md                   # Project documentation
├── main.py                     # Main entry point
├── config/                     # Configuration files
│   └── default.json
├── src/                        # Source code (for larger projects)
│   ├── __init__.py
│   └── modules.py
├── tests/                      # Project-specific tests
│   └── test_main.py
└── docs/                       # Additional documentation
    └── usage.md
```

## Development Workflow

### 1. Before Making Changes
```bash
# Update your local repository
git pull origin main

# Create a feature branch
git checkout -b feature/your-feature-name

# Run tests to ensure current state
python -m pytest tests/
```

### 2. Code Quality Standards

#### Formatting with Black
```bash
# Format all Python files
black .

# Check what would be formatted
black --check .
```

#### Linting with Flake8
```bash
# Lint all Python files
flake8 .

# Lint specific project
flake8 project_name/
```

#### Type Checking with MyPy (for larger projects)
```bash
# Type check specific files
mypy shared_utils.py
mypy project_name/main.py
```

### 3. Testing Guidelines

#### Running Tests
```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=. --cov-report=html

# Run tests for specific project
python -m pytest tests/test_project_name.py
```

#### Writing Tests
Create test files following the pattern `test_<module_name>.py`:

```python
import pytest
from shared_utils import ProjectConfig

def test_config_initialization():
    """Test that configuration initializes properly."""
    config = ProjectConfig()
    assert config is not None
    assert isinstance(config.config, dict)

def test_safe_file_operation():
    """Test file operations work correctly."""
    from shared_utils import safe_file_operation
    
    # Test file existence check
    result = safe_file_operation('exists', 'nonexistent_file.txt')
    assert result is False
```

### 4. Documentation Standards

#### Code Documentation
- Use Google-style docstrings
- Include type hints for function parameters and returns
- Document complex logic with inline comments

```python
def process_data(input_file: str, output_format: str = "json") -> Optional[Dict[str, Any]]:
    """
    Process data from input file and return formatted results.
    
    Args:
        input_file: Path to the input file
        output_format: Format for output data (json, csv, txt)
    
    Returns:
        Processed data dictionary or None if processing failed
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If output_format is not supported
    """
    pass
```

#### README Structure
Each project should have a README with:
1. Brief description and purpose
2. Installation requirements
3. Usage examples
4. Configuration options
5. API documentation (if applicable)

## Working with Specific Project Types

### AI/LLM Projects
- Ensure API keys are configured in environment variables
- Use `shared_utils.RateLimiter` for API calls
- Test with mock responses when possible

### File Processing Projects
- Use `shared_utils.safe_file_operation` for file I/O
- Handle various file encodings
- Provide examples with sample data

### GUI Projects
- Document system requirements (PyQt5, tkinter)
- Include screenshots in documentation
- Test on multiple platforms when possible

### Game Projects
- Include save/load functionality testing
- Document game rules and mechanics
- Provide example gameplay scenarios

## Common Issues and Solutions

### Dependency Issues
```bash
# If import errors occur, check dependencies
pip list | grep package_name

# Reinstall specific package
pip uninstall package_name
pip install package_name

# Update all packages
pip install --upgrade -r requirements.txt
```

### API Key Issues
```bash
# Check if environment variables are set
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY

# Set temporarily for testing
export ANTHROPIC_API_KEY="your_key_here"
```

### Path Issues
- Use `pathlib.Path` for cross-platform compatibility
- Use relative paths from project root
- Check current working directory in scripts

## Contributing Guidelines

### Pull Request Process
1. Ensure all tests pass
2. Update documentation for any new features
3. Follow the existing code style
4. Include tests for new functionality
5. Update TECHNICAL_EVALUATION.md if adding new dependencies

### Code Review Checklist
- [ ] Code follows style guidelines (black, flake8)
- [ ] New functions have proper docstrings
- [ ] Tests cover new functionality
- [ ] Documentation is updated
- [ ] No hardcoded credentials or paths
- [ ] Error handling is appropriate
- [ ] Logging uses standard format

### Performance Considerations
- Use generators for large data processing
- Implement proper caching where applicable
- Add async support for I/O-bound operations
- Profile code for bottlenecks in complex projects

## Debugging Tips

### Common Debugging Techniques
```python
# Use logging instead of print statements
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
```

### IDE Setup
Recommended IDE configurations:
- **VS Code**: Install Python, Black Formatter, and Flake8 extensions
- **PyCharm**: Configure Black as external tool, enable Flake8 inspection
- **Vim/Neovim**: Use ALE plugin with Black and Flake8

### Environment Debugging
```bash
# Check Python version and path
python --version
which python

# Check installed packages
pip list

# Check environment variables
env | grep API

# Check current directory and path
pwd
echo $PYTHONPATH
```

## Resources

### Documentation
- [Python Style Guide](https://peps.python.org/pep-0008/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Flake8 Linting](https://flake8.pycqa.org/)
- [Pytest Testing](https://docs.pytest.org/)

### Project-Specific Resources
- **Anthropic API**: [Documentation](https://docs.anthropic.com/)
- **OpenAI API**: [Documentation](https://platform.openai.com/docs/)
- **PyQt5**: [Documentation](https://doc.qt.io/qtforpython/)
- **Flask**: [Documentation](https://flask.palletsprojects.com/)

## Getting Help

1. **Check existing documentation** in project README files
2. **Review similar projects** in the repository for patterns
3. **Check TECHNICAL_EVALUATION.md** for known issues and solutions
4. **Create an issue** on GitHub for bugs or feature requests
5. **Start a discussion** for questions about architecture or best practices

---

*This guide is updated regularly. Last updated: 2024-08-23*