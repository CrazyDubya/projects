# inefficientwatcher.py
import time


def extract_extension(first_line):
    comment_markers = {
        '#': 'Python, Shell',
        '//': 'JavaScript, Java, C++, Go',
        '/*': 'JavaScript, C++, CSS, Java',
        '--': 'SQL',
        '<!--': 'HTML'
    }
    for marker, languages in comment_markers.items():
        if first_line.strip().startswith(marker):
            start_index = first_line.find(marker) + len(marker)
            extension = first_line[start_index:].strip()
            return extension, marker  # Return both extension and marker for logging
    return None, None


def convert_files(directory):
    print(f"Scanning directory: {directory}")
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            try:
                print(f"Processing file: {file_path}")
                with open(file_path, 'r') as file:
                    first_line = file.readline()
                    print(f"First line of the file: {first_line.strip()}")
                    extension, marker = extract_extension(first_line)
                    print(f"Extracted extension: {extension}, marker: {marker}")
                    if extension:
                        new_filename = os.path.splitext(filename)[0] + '.' + extension
                        torun_dir = os.path.join(directory, 'torun')
                        os.makedirs(torun_dir, exist_ok=True)  # Create 'torun' directory if it doesn't exist
                        new_path = os.path.join(torun_dir, new_filename)
                        print(f"Renaming file from: {file_path} to: {new_path}")
                        os.rename(file_path, new_path)
                        print(f"File renamed to: {new_path} using marker: {marker}")
                        # Call PyCharm to run the script
                        print(f"Calling PyCharm to run the script: {new_path}")
                        os.system(f"pycharm {new_path}")
                    else:
                        print(
                            f"No valid marker found in the first line: '{first_line.strip()}' or incorrect file structure.")
            except PermissionError as pe:
                print(f"Permission denied while processing file: {file_path}")
                print(f"Error details: {pe}")
            except Exception as e:
                print(f"Error handling file: {file_path}")
                print(f"Error details: {e}")


def log_monitoring_activity(directory):
    """Logs the script's monitoring activity to a file in the directory."""
    file_path = os.path.join(directory, 'watching2.txt')
    try:
        with open(file_path, 'a') as f:
            f.write(f"Checked at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")


if __name__ == "__main__":
    path_to_watch = '/Users/puppuccino/Library/Mobile Documents/com~apple~CloudDocs/PyCharmTransfer'
    print(f"Starting to monitor directory: {path_to_watch}")
    try:
        while True:
            print("Logging monitoring activity...")
            log_monitoring_activity(path_to_watch)
            convert_files(path_to_watch)
            print(f"Waiting for 30 seconds before the next scan...")
            time.sleep(30)
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping the script...")
