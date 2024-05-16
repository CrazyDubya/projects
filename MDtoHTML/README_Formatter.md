
# Markdown Formatter

## Overview
`formatter.py` is a Python script that converts user-provided Markdown content into HTML and displays it in the default web browser. The script uses a simple dialog for user input and handles the conversion and preview process seamlessly.

## Features
- **Markdown to HTML Conversion**: Converts Markdown content to HTML.
- **HTML Preview in Browser**: Opens the converted HTML content in the default web browser.
- **Temporary File Handling**: Uses a temporary HTML file to display the content.
- **User Input**: Prompts the user to enter Markdown content through a simple dialog.

## Usage
1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install Dependencies**:
    ```bash
    pip install markdown2
    ```

3. **Run the Script**:
    ```bash
    python formatter.py
    ```

4. **Interact with the Script**:
   - A dialog will appear prompting you to enter Markdown content.
   - The script will convert the entered Markdown to HTML and open it in your default web browser for preview.

## Example
```bash
$ python formatter.py
```
- A dialog box appears prompting you to paste your Markdown content.
- After pasting and confirming, the content is converted to HTML and displayed in your web browser.

## Configuration
No additional configuration is required.

## Future Expansion Plans
1. **Enhanced Input Validation**: Improve validation for Markdown input.
2. **Customization Options**: Allow users to customize the HTML template and CSS styling.
3. **Additional Output Formats**: Support other output formats like PDF.
4. **Advanced Features**: Implement features like live preview and Markdown syntax highlighting.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License
This project is licensed under the MIT License.
