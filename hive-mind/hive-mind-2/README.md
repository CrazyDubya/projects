
# Hive Mind Manager

## Overview
`hive-mind2.py` is a Python script designed to manage a hive mind structure, facilitating communication between nodes with the assistance of an AI model (Claude). It includes features for metadata prompt management, node communication, XML handling, logging, and a graphical interface for interaction.

## Features
- **MetaPrompt Management**: Define and manage metadata prompts, including creation, modification, retrieval, and rating updates.
- **Node Communication**: Facilitate communication between nodes in a hive mind with Claude as the AI assistant.
- **XML Handling**: Generate and process XML content for node communication and state persistence.
- **Logging**: Track node activities and errors using logging.
- **HiveMind Coordination**: Manage hive mind structure, node creation, and communication.
- **GUI Interface**: Provide a graphical interface for setting hive mind goals, viewing leader output, node communication, and progress tracking.

## Usage
1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install Dependencies**:
    ```bash
    pip install anthropic PyQt5
    ```

3. **Run the Script**:
    ```bash
    python header.py
    ```

4. **Interact with the GUI**:
   - Set the hive mind goal using the provided input field.
   - Load and view leader output, node lists, and node communication from files.
   - Track and display progress updates.

## Example
```bash
$ python header.py
```
- A graphical interface appears, allowing you to set goals, load files, and view outputs.

## Configuration
- **nodes_directory**: Set the path to the directory containing the nodes. Modify the `nodes_directory` variable in the script as needed.

## Future Expansion Plans
1. **Enhanced MetaPrompt Management**: Improve the functionality for managing metadata prompts.
2. **Advanced Node Communication**: Implement more sophisticated communication mechanisms between nodes.
3. **Additional XML Features**: Extend XML handling to include more node attributes and interactions.
4. **Improved GUI**: Add more features and customization options to the graphical interface.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License
This project is licensed under the MIT License.
