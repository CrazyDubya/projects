# header.py
import re

def update_file_header(path):
    # Define the expected pattern
    pattern = r"^# .+\.py$"

    # Read the contents of the file
    with open(path, 'r+') as file:
        lines = file.readlines()
        if not lines:
            return  # Skip empty files
        first_line = lines[0].strip()

        # Check if the first line is already correct
        expected_header = f"# {os.path.basename(path)}"
        if re.match(pattern, first_line):
            return  # No update needed if the pattern matches

        # Update the first line if it doesn't match
        lines[0] = expected_header + '\n'
        file.seek(0)
        file.writelines(lines)
        file.truncate()

def process_directory(directory):
    # Walk through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.py'):
            update_file_header(os.path.join(directory, filename))

if __name__ == '__main__':
    # Set the directory path
    directory_path = '/Users/puppuccino/PycharmProjects/inner_mon/'
    process_directory(directory_path)
    print("Header checks and updates are complete.")
