
# Move Log Files

## Overview
`movelog.py` is a Python script designed to move log and text files from the current working directory to their respective directories. The script checks for the existence of the `log` and `txt` directories and creates them if they do not exist.

## Features
- **Move Log Files**: Move all `.log` files from the current working directory to the `log` directory.
- **Move Text Files**: Move all `.txt` files from the current working directory to the `txt` directory.
- **Directory Creation**: Automatically create the `log` and `txt` directories if they do not exist.

## Usage
1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Run the Script**:
    ```bash
    python movelog.py
    ```

## Example
```bash
$ python movelog.py
```
- The script will move all `.log` files to the `log` directory and all `.txt` files to the `txt` directory.

## Configuration
- **directory**: Set to the current working directory by default. Modify as needed to specify a different directory.

## Future Expansion Plans
1. **Enhanced File Handling**: Add support for moving other file types.
2. **Recursive Directory Processing**: Process files in subdirectories.
3. **Customization Options**: Provide more options for customizing the destination directories.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License
This project is licensed under the MIT License.
