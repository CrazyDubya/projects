
# Obsidian Note Generator

## Overview
`obsidian_note_generator.py` is a Python script designed to create and move Obsidian notes from Python files. It reads the content of each Python file in a specified directory, creates a corresponding Markdown file with the content, and moves the Markdown file to an Obsidian vault.
'texttomd.py" renames txt files to md

## Features
- **Create Obsidian Notes**: Generate Markdown files from Python files.
- **Move Notes to Obsidian Vault**: Move the generated Markdown files to a specified Obsidian vault.
- **Directory Processing**: Process all Python files in a specified directory.

## Usage
1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Run the Script**:
    ```bash
    python obsidian_note_generator.py
    ```

3. **Configuration**:
   - **directory_path**: Set the path to the directory containing your Python files.
   - **obsidian_vault_path**: Set the path to your Obsidian vault.

## Example
```bash
$ python obsidian_note_generator.py
```
- The script will create Markdown notes for each Python file in the specified directory and move them to the Obsidian vault.

## Configuration
- **directory_path**: Set the path to the directory containing your Python files.
- **obsidian_vault_path**: Set the path to your Obsidian vault in iCloud.

## Future Expansion Plans
1. **Enhanced Note Generation**: Improve the formatting and content of the generated notes.
2. **Recursive Directory Processing**: Add support for processing files in subdirectories.
3. **Customization Options**: Provide more options for customizing the generated notes.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License
This project is licensed under the MIT License.
