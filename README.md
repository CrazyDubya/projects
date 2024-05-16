# allseeingeye
AllSeeingEye

Overview

AllSeeingEye is a Python script designed to recursively list the contents of a directory, build a visual tree of the directory structure, and process files based on their types. The script generates output files that include or exclude the contents of text files, providing a comprehensive view of the directory's structure and contents.

Features

Directory Listing: Uses the ls command to list directory contents, including hidden files.
File Processing:
Processes files differently based on their extensions.
Includes the content of code and data files (.py, .js, .html, .css, .csv, .txt) in the output.
Lists the name and location of other files.
Handles potential errors in reading files, such as permission issues.
Directory Tree Building:
Recursively builds a visual tree of the directory structure.
Lists contents and processes files based on their type.
Skips certain directories (chatter, data) and notes their purpose.
Excludes specific files (chatter/data/chat.html, chatter/data/conversations.json) from processing.
Main Function:
Runs the tree command and outputs the result to a temporary file.
Builds the directory structure and file content into two separate output files:
One including text file contents.
One excluding text file contents.
Cleans up the temporary tree output file.
Outputs the final results.
Usage

Clone the repository:

bash
Copy code
git clone <repository-url>
cd <repository-directory>
Run the script:

bash
Copy code
python allseeingeye.py
Output:

directory_structure_and_files.txt: Contains the directory tree and file contents.
directory_structure_and_files_no_text.txt: Contains the directory tree without text file contents.
Future Expansion Plans

Cross-Platform Compatibility: Modify the script to support Windows systems.
Customization Options: Allow users to customize the types of files processed and their handling.
Error Handling Enhancements: Improve error handling to provide more detailed error messages and recovery options.
Output Formats: Add support for different output formats, such as JSON or HTML.
GUI Interface: Develop a graphical user interface for easier interaction with the script.
Performance Optimization: Optimize the script for faster processing of large directories.
Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

License

This project is licensed under the MIT License.

