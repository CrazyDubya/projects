# txttomd.py
import shutil


def copy_and_convert_txt_to_md(source_dir, target_dir):
    # Walk through all subdirectories and files in the source directory
    for root, dirs, files in os.walk(source_dir):
        # Process each file in the current directory
        for file in files:
            if file.endswith('.txt'):
                source_file_path = os.path.join(root, file)
                # Construct the target path by replacing the source directory with the target directory in the path
                target_file_path = source_file_path.replace(source_dir, target_dir)
                target_file_path = os.path.splitext(target_file_path)[0] + '.md'  # Change the extension to .md

                # Ensure the target directory exists
                os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

                # Copy the file and rename to .md
                shutil.copy(source_file_path, target_file_path)


if __name__ == '__main__':
    # Define the source and target directories
    source_directory = '/Users/puppuccino/PycharmProjects/inner_mon/brainstorm'
    target_directory = '/Users/puppuccino/Documents/Obsidian Vault/Brainstorm'

    # Execute the function to copy and convert files
    copy_and_convert_txt_to_md(source_directory, target_directory)
    print("All text files have been copied and converted to markdown files in the Obsidian vault.")
