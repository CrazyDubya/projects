# allseeingeye.py
import subprocess

# Define the types of files to process differently
code_files = ('.py', '.js', '.html', '.css', '.csv')
data_files = (".txt", )

def list_directory(directory):
    """ Uses ls to list the directory contents, including hidden files. This function is Unix/Linux/MacOS specific. """
    cmd = ['ls', '-1a', directory]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    return result.stdout.splitlines()

def process_file(file_path, output_file, include_text=True):
    """Process each file based on its extension."""
    _, ext = os.path.splitext(file_path)
    try:
        if ext in code_files or (ext in data_files and include_text):
            # For code and data files, include the content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            header = f"## {os.path.basename(file_path)} ##\n" if ext in code_files else ""
            output_file.write(f"{header}{content}\n\n")
        else:
            # For other files, just list the file name and location
            output_file.write(f"{file_path}\n")
    except Exception as e:
        # Handle potential errors in reading files, e.g., permission issues
        output_file.write(f"Error processing file {file_path}: {e}\n")

def build_tree(directory, output_file, include_text, prefix=''):
    """ Recursively builds a visual tree of the directory structure, lists contents, and processes files based on their type. """
    entries = list_directory(directory)
    for entry in entries:
        if entry in ['.', '..']:
            continue  # Skip the current and parent directory markers
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            # Directory: write its name and explore it recursively
            if entry not in ['chatter', 'data']:
                output_file.write(f"{prefix}+-- {entry}/\n")
                build_tree(path, output_file, include_text, prefix=prefix + " ")
            else:
                output_file.write(f"{prefix}+-- {entry}/ # Contains full archive of user conversations with ChatGPT\n")
        else:
            # File: process it based on its type
            if path != 'chatter/data/chat.html' and path != 'chatter/data/conversations.json':
                output_file.write(f"{prefix}+-- {entry}\n")
                process_file(path, output_file, include_text)

def main():
    tree_output_filename = "_directory_tree.txt"
    final_output_filename = "directory_structure_and_files.txt"
    final_output_filename_no_text = "directory_structure_and_files_no_text.txt"
    current_directory = os.getcwd()

    # Run the tree command and output the result to a temporary file
    with open(tree_output_filename, "w") as tree_output_file:
        subprocess.run(["tree", "-a", current_directory], stdout=tree_output_file, text=True)

    # Now, build the directory structure and files content
    for include_text, output_filename in [(True, final_output_filename), (False, final_output_filename_no_text)]:
        with open(output_filename, "w") as output_file:
            # First, write the tree command output
            with open(tree_output_filename, "r") as tree_output_file:
                output_file.write(tree_output_file.read())

            # Then, append the detailed directory structure and files content
            build_tree(current_directory, output_file, include_text)

    # Cleanup the temporary tree output file if needed
    os.remove(tree_output_filename)

    print(f"Combined directory tree and structure with files have been written to {final_output_filename}")
    print(f"Combined directory tree and structure without .txt file contents have been written to {final_output_filename_no_text}")

if __name__ == "__main__":
    main()
