
# File Mover Script

## Overview
`mover.py` is a Python script designed to monitor a source directory for new `.txt` files and move them to a specified destination directory. The script ensures that the destination directory exists and creates it if it does not.

## Features
- **Monitor Source Directory**: Continuously monitor a source directory for new `.txt` files.
- **Move Files**: Move new `.txt` files from the source directory to the destination directory.
- **Create Destination Directory**: Ensure the destination directory exists and create it if it does not.

## Usage
1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Modify Source and Destination Directories**:
   - Set the `src_dir` variable to the path of the source directory to monitor.
   - Set the `dst_dir` variable to the path of the destination directory where files will be moved.

3. **Run the Script**:
    ```bash
    python mover.py
    ```

## Example
```bash
$ python mover.py
```
- The script will monitor the source directory for new `.txt` files and move them to the destination directory.

## Configuration
- **src_dir**: Set the path to the source directory to monitor.
- **dst_dir**: Set the path to the destination directory where files will be moved.

## Future Expansion Plans
1. **Enhanced File Monitoring**: Add support for monitoring multiple file types.
2. **Error Handling**: Implement robust error handling for file operations.
3. **Notification System**: Add notifications for file movements.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License
This project is licensed under the MIT License.
