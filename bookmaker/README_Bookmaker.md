
# Script Runner

## Overview
The Script Runner project consists of multiple scripts designed to work together for monitoring directories, processing files, and automating tasks. The key scripts include `_scriptrunner.py`, `mover.py`, `WatchCharm.py`, and `autopy.py`.

## Features
### _scriptrunner.py
- **Script Runner**: Runs multiple Python scripts concurrently.
- **Process Management**: Starts all specified scripts and waits for them to complete.

### mover.py
- **File Monitoring**: Monitors a specified source directory for new `.txt` files.
- **File Moving**: Moves new `.txt` files to a specified destination directory.

### WatchCharm.py
- **File Monitoring**: Monitors a specified directory for new files.
- **File Processing**:
  - Extracts file extensions from the first line of `.txt` files.
  - Moves files to appropriate directories based on extracted extensions.
  - Handles files without valid extensions and moves them to a 'processed' directory.

### autopy.py
- **File Monitoring**: Monitors a specified directory for new `.py` files.
- **Script Execution**: Automatically runs new Python scripts upon detection.
- **Real-Time Logging**: Captures console output in real-time and saves it to a log file.
- **Desktop Image Capture**: Takes a screenshot of the desktop after running the script.

## Usage
1. **Set Up Environment**:
   - Ensure all required dependencies are installed.
   - Modify paths in the scripts to suit your directory structure.

2. **Run _scriptrunner.py**:
    ```bash
    python _scriptrunner.py
    ```

3. **Monitor Directories**:
   - The scripts will automatically monitor and process files in the specified directories.

## Future Expansion Plans
1. **Enhanced Error Handling**: Improve error messages and provide options for recovery.
2. **Customizable Configurations**: Allow users to configure directories, file types, and other settings.
3. **Additional File Types**: Extend support for more file types and processing rules.
4. **Advanced Logging**: Implement more sophisticated logging mechanisms.
5. **Performance Optimization**: Optimize the scripts for faster and more efficient file processing.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License
This project is licensed under the MIT License.
