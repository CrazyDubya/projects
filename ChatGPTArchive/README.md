
# chatgptarchive.py

## Overview

`chatgptarchive.py` is a Python script designed to parse conversations from a JSON file (e.g., a ChatGPT conversation archive) and save them to individual JSON files. This tool is helpful for managing and analyzing conversation data more efficiently.

## Prerequisites

Before you run `chatgptarchive.py`, you need to have Python installed on your system. This script is compatible with Python 3.6 or later.

## Getting Your ChatGPT Conversation Archive

### Requesting the Archive

1. **OpenAI Users:** Log in to your ChatGPT account, navigate to settings/data controls, and request your data. The data will delivered to you via email as a downloadable zip file.

### Unzipping the Archive

- **On macOS:**
  1. Open Terminal.
  2. Navigate to the directory containing the downloaded zip file.
  3. Use the command `unzip [filename].zip` - replace `[filename]` with the name of your downloaded file.

- **On Windows:**
  1. Navigate to the folder containing the zip file in File Explorer.
  2. Right-click on the zip file.
  3. Select "Extract All..." and follow the instructions to extract the files.

- **On Linux:**
  1. Open Terminal.
  2. Navigate to the directory containing the downloaded zip file.
  3. Use the command `unzip [filename].zip`.

## Installation

No additional libraries are required for this script. It uses only the built-in `json`, `os`, and `re` modules available in Python's standard library.

## Usage

To run `chatgptarchive.py`:

1. Open your command line interface (Terminal on macOS and Linux, Command Prompt or PowerShell on Windows).
2. Navigate to the directory containing `chatgptarchive.py`.
3. Run the script using the command:
   ```
   python chatgptarchive.py
   ```
4. When prompted, enter the full path to your unzipped `conversations.json` file.
5. The script will process the data and save each conversation as a separate JSON file in a directory named `gptlogs-DDMMYY` (with the current date) within the same directory as your `conversations.json`.

## Output

After running the script, each conversation from the `conversations.json` will be saved as an individual JSON file in the `gptlogs-DDMMYY` directory. These files are named based on the title of the conversation or assigned a default name if the title is not available.

## Troubleshooting

If you encounter any issues regarding file paths or permissions, ensure that the path you input corresponds exactly to where your `conversations.json` file is located and that you have appropriate read/write permissions for the directory.

---

This README provides a comprehensive guide to obtaining, preparing, and processing your ChatGPT conversation data using `chatgptarchive.py`.

### analyze_conversations.py

This script analyzes the conversations to find common words and perform sentiment analysis.

### chatgptreader.py
This script helps you load and format a conversation from the saved JSON files.

