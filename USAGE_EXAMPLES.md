# Project Usage Examples

This document provides practical examples of how to use the various projects in this repository.

## Quick Start

```bash
# Install all dependencies
pip install -r requirements.txt

# Validate all projects
python validate_projects.py

# Get an overview of all projects
cat INDEX.md
```

## 🧮 Mathematical Computing (pi)

Calculate PI with high precision using multiple algorithms:

```bash
cd pi/python

# Basic PI calculation (100 digits)
python pi.py 100

# Gauss-Legendre with custom iterations
python pi2.py 50 8

# Compare multiple algorithms with parallel processing
python py-multi.py 100 8 1000000
```

**Output:** High-precision PI calculations saved to individual result files.

## 🤖 AI & LLM Tools

### ChatGPT Archive Analysis
```bash
cd ChatGPTArchive

# Analyze conversation data
python chatgptarchive.py conversations.json

# Generate word cloud from conversations
python gptwordcloud-2.py conversations.json

# Advanced stats with Claude integration
cd StatsplusClaude
python Claude-chat-gptarchive-public.py ../conversations.json
```

### Multi-LLM Conversations
```bash
cd llmchatroom

# Set up API keys
export OPENAI_API_KEY="your_key_here"
export CLAUDE_API_KEY="your_key_here"

# Start multi-LLM conversation
python llmchatroom.py
```

### Anthropic API Interface
```bash
cd ant

# Set API key
export CLAUDE_API_KEY="your_key_here"

# Start conversation
python ant.py
```

## 🎮 Games & Simulations

### 4X Space Simulation
```bash
cd 4x

# Start colony management
python colony_management.py

# Run galactic events
python gal_event_man.py

# Design ships
python ship_design.py

# Manage diplomatic relations
python civ_dip.py
```

### Quantum Chess
```bash
cd Quantum_Chess

# Start quantum chess game
python QC.py
```

### Self-Modifying Game (Nomic)
```bash
cd nomic

# Start nomic game with AI players
python nomic.py
```

## 🔧 File Processing & Utilities

### Directory Analysis
```bash
cd allseeingeye

# Analyze current directory with content
python allseeingeye.py

# Results saved to:
# - directory_structure_and_files.txt (with content)
# - directory_structure_and_files_no_text.txt (structure only)
```

### JSON Processing
```bash
cd jsonreader

# Process JSON file
python jsonreader.py data.json

# Alternative processor
python jsonreader2.py data.json
```

### XML Merging
```bash
cd xmlmerge

# Merge multiple XML files
python xmlmerger.py file1.xml file2.xml file3.xml
```

### File Format Conversion
```bash
cd MDtoHTML
python formatter.py document.md

cd MakeMarkdown
python txttomd.py document.txt

cd HeaderPy
python header.py source_file.py
```

## 🔄 System Automation

### File Movement
```bash
cd mover

# Monitor and move .txt files
python mover.py

cd movelog

# Organize log files
python movelog.py
```

### Cross-Device File Transfer
```bash
cd "iPhone toss to Mac"

# Monitor directory for new files
python WatchCharm.py

# Run multiple scripts concurrently
python scriptrunner.py

# Auto-execute Python files
python autopy.py
```

## 💬 Communication Tools

### Chatroom
```bash
cd chatroom

# Start GUI chatroom
python chatroom.py
```

### Bluetooth Communication
```bash
cd bluetooth

# Discover Bluetooth devices
python bluetooth.py

# Print to Bluetooth device
python bluetoothprint.py

# Find specific Bluetooth printer
python buetoothprintfind.py
```

## 🤝 Advanced AI Systems

### Hive-Mind Distributed Processing
```bash
cd hive-mind

# Start main hive-mind system
python hive-mind/hive-mind.py

# Alternative implementation
python hivemind2.py

# GUI version
python Z/zhive-mind.py
```

### Inner Monologue Processing
```bash
cd inner_monologue

# Set API key
export CLAUDE_API_KEY="your_key_here"

# Start inner monologue system
python inner_monologue.py

# GUI version
python inn_mono_gui.py
```

### Content Generation
```bash
cd brainstorm

# Set API keys
export CLAUDE_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"

# Start brainstorming session
python brainstorm.py

cd bookmaker

# Generate book content
python bookmaker.py
```

## 📊 Environment Setup for AI Projects

Most AI projects require API keys. Set them up as follows:

```bash
# Anthropic (Claude)
export CLAUDE_API_KEY="sk-ant-your-key-here"

# OpenAI (GPT models)
export OPENAI_API_KEY="sk-your-key-here"

# Google (Gemini)
export GOOGLE_API_KEY="your-key-here"

# Monster API
export MONSTER_API_KEY="your-key-here"

# Perplexity API
export PERPLEXITY_API_KEY="your-key-here"
```

## 🔍 Troubleshooting

### Common Issues

1. **Missing Dependencies**: Run `pip install -r requirements.txt`
2. **API Key Errors**: Ensure environment variables are set correctly
3. **File Permissions**: Make scripts executable with `chmod +x script.py`
4. **Import Errors**: Check if you're in the correct directory

### Validation

Use the validation script to check project health:

```bash
python validate_projects.py
```

### Getting Help

1. Check individual project README files for specific instructions
2. Review the `INDEX.md` and `PROJECT_INDEX.md` for project capabilities
3. Examine the source code for usage patterns and examples

## 🎯 Project Categories Quick Reference

- **🤖 AI/LLM Tools**: ant, brainstorm, bookmaker, ChatGPTArchive, chatter, chatroom, hive-mind, inner_monologue, llmchatroom, noder
- **🎮 Games**: 4x, nomic, Quantum_Chess
- **🔧 Utilities**: allseeingeye, HeaderPy, jsonreader, MDtoHTML, MakeMarkdown, xmlmerge
- **🔄 Automation**: iPhone toss to Mac, mover, movelog
- **💬 Communication**: bluetooth, chatroom
- **📊 Data/Math**: pi, movelog
- **🎬 Media**: ffmpeg-gui

---

*For the most up-to-date information, always refer to individual project documentation and the main INDEX.md file.*