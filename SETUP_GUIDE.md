# Repository Setup and Installation Guide

## 🚀 Quick Start

This repository contains 22 diverse Python projects. Follow this guide to set up your environment and start using the tools.

### System Requirements
- **Python**: 3.7+ (recommended: 3.9+)
- **Operating System**: Cross-platform (Linux/macOS/Windows)
- **Memory**: 2GB+ RAM recommended for AI/ML projects
- **Storage**: 1GB+ free space

## 📦 Installation Options

### Option 1: Full Repository Setup (Recommended)
Install all dependencies for complete functionality:

```bash
# Clone the repository
git clone https://github.com/CrazyDubya/projects.git
cd projects

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements_all.txt
```

### Option 2: Individual Project Setup
Install dependencies only for specific projects you need:

```bash
# For AI/ML projects
pip install anthropic openai transformers torch sklearn

# For GUI applications  
pip install tkinter PyQt5

# For data processing
pip install lxml numpy matplotlib wordcloud textstat textblob

# For specialized tools
pip install bleak watchdog networkx prometheus_client rich
```

### Option 3: Core Dependencies Only
Minimal installation for basic functionality:

```bash
pip install requests json pathlib os sys
```

## 🎯 Project Categories and Setup

### 🤖 AI & Language Model Tools
**Required API Keys**: Configure these as environment variables
```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export GOOGLE_API_KEY="your_google_key"  # For Gemini models
```

**Key Projects**:
- `llmchatroom/` - Multi-LLM conversation facilitator
- `ChatGPTArchive/` - ChatGPT conversation analysis
- `ant/` - Anthropic API interface
- `inner_monologue/` - Internal thought processing

**Setup**:
```bash
cd llmchatroom
python llmchatroom.py
```

### 🎮 Games & Simulations
**Projects**:
- `4x/` - Space simulation game suite (728 LOC)
- `Quantum_Chess/` - Quantum mechanics chess variant
- `nomic/` - Self-modifying rule game

**Setup**:
```bash
cd 4x
python colony_management.py  # Start with colony management
```

### 📁 File Management & Processing
**Projects**:
- `allseeingeye/` - Directory analysis tool
- `mover/` - Automated file movement
- `jsonreader/` - JSON processing tool
- `xmlmerge/` - XML file merging

**Setup**:
```bash
cd allseeingeye
python allseeingeye.py  # Analyze any directory
```

### 🔧 Automation & Utilities
**Projects**:
- `bookmaker/` - Document generation
- `noder/` - Node-based processing
- `iPhone toss to Mac/` - Cross-device file transfer

**Setup**:
```bash
cd bookmaker
python bookmaker.py
```

## 🔧 Advanced Configuration

### API Configuration
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
MONSTER_API_KEY=your_monster_key_here
```

### Project-Specific Configuration
Some projects require additional setup:

#### ChatGPTArchive
```bash
cd ChatGPTArchive
# Place your conversations.json file here
python chatgptarchive.py
```

#### Bluetooth Communication
```bash
# Additional setup for Bluetooth projects
pip install bleak
# Ensure Bluetooth is enabled on your system
```

## 📊 Repository Analysis Tools

### Run Comprehensive Analysis
```bash
# Analyze all projects for metrics and health
python repository_analyzer.py

# Generate enhanced documentation
python documentation_expander.py
```

### View Analysis Results
- `repository_analysis.json` - Raw analysis data
- `INDEX_ENHANCED.md` - Enhanced repository overview
- `REPOSITORY_HEALTH.md` - Project health dashboard

## 🚨 Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors for harmonized_api_wrappers
cd bookmaker
# Ensure the file exists and dependencies are installed
```

#### API Key Issues
```bash
# Verify your API keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

#### Permission Issues (Unix/Linux)
```bash
# Make scripts executable
chmod +x *.py
```

### Platform-Specific Notes

#### Windows
- Use `venv\Scripts\activate` instead of `source venv/bin/activate`
- Some file paths may need adjustment for Windows directory separators

#### macOS
- You may need to install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

#### Linux
- Ensure Python 3.7+ is installed:
  ```bash
  sudo apt update
  sudo apt install python3.9 python3.9-venv python3-pip
  ```

## 📈 Performance Optimization

### For Large Projects (4x, ChatGPTArchive)
- Allocate at least 4GB RAM
- Use SSD storage for better I/O performance
- Consider running on Python 3.9+ for performance improvements

### For AI/ML Projects
- GPU acceleration recommended for torch-based projects
- Increase timeout values for API calls if needed
- Monitor API rate limits

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install black flake8 pytest mypy

# Run code formatting
black *.py

# Run linting
flake8 *.py

# Run type checking
mypy *.py
```

### Testing
```bash
# Run repository analysis to verify setup
python repository_analyzer.py

# Test individual projects
cd allseeingeye && python allseeingeye.py
```

## 📝 Next Steps

1. **Choose your focus area**: AI tools, games, utilities, or file processing
2. **Set up API keys** for AI/ML projects
3. **Run the analysis tools** to understand the codebase
4. **Start with smaller projects** like `allseeingeye` or `mover`
5. **Explore integration opportunities** between related projects

## 📞 Support

- Check the `README.md` files in individual project directories
- Review the enhanced documentation in `INDEX_ENHANCED.md`
- Analyze project health using `REPOSITORY_HEALTH.md`
- Run `python repository_analyzer.py` for up-to-date metrics

---

*This guide was generated based on automated analysis of the repository structure and dependencies.*