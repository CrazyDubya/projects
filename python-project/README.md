# Python Parser Project

A Python project focused on parsing and agent functionality.

## Features
- OpenAI/Azure OpenAI integration with usage logging
- Agent-based processing
- Parsing utilities

## Setup
1. Create virtual environment: `python -m venv .venv`
2. Activate: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

## Usage
Run the main parser script to process data with AI assistance.

## Files
- `agent.py` - Main agent functionality
- `parsing.py` - Data parsing utilities

## Interactive Solar System
This project now includes an interactive 2D simulation of our solar system, built with HTML, CSS, and JavaScript. It visually represents the Sun and the planets of our solar system, with planets orbiting the Sun at relatively representative speeds.

### How to View
1. Ensure you have all the project files, particularly those in the `solar_system/` directory.
2. Open the `solar_system/index.html` file directly in your web browser (e.g., Chrome, Firefox, Safari, Edge).
   - You can typically do this by navigating to the file in your file explorer, right-clicking it, and selecting "Open with" your preferred browser.
   - Alternatively, you can type the full file path into your browser's address bar (e.g., `file:///path/to/your/repository/solar_system/index.html`).

The simulation should load, and you will see the planets orbiting the central sun. Basic tests for the simulation can be found in `solar_system/test.html`, which can also be opened in a browser to view test results (requires internet for QUnit CDN).
