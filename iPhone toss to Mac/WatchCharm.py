# WatchCharm.py
import shutil

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


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


class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        print(f"New file or directory created: {event.src_path}")
        if not event.is_directory and event.src_path.endswith('.txt'):
            try:
                with open(event.src_path, 'r') as file:
                    first_line = file.readline()
                print(f"First line of the file: {first_line.strip()}")
                extension, marker = extract_extension(first_line)
                print(f"Extracted extension: {extension}, marker: {marker}")

                dirname, basename = os.path.split(event.src_path)

                if extension:
                    new_filename = os.path.splitext(basename)[0] + '.' + extension
                    charmed_dir = os.path.join(dirname, 'charm')
                    os.makedirs(charmed_dir, exist_ok=True)  # Create 'charmed' directory if it doesn't exist
                    new_path = os.path.join(charmed_dir, new_filename)
                    print(f"Moving file from: {event.src_path} to: {new_path}")
                    shutil.move(event.src_path, new_path)
                    print(f"File moved to: {new_path} using marker: {marker}")
                else:
                    print(
                        f"No valid marker found in the first line: '{first_line.strip()}' or incorrect file structure.")
                    processed_dir = os.path.join(dirname, 'processed')
                    os.makedirs(processed_dir, exist_ok=True)  # Create 'processed' directory if it doesn't exist
                    processed_path = os.path.join(processed_dir, basename)
                    print(f"Moving processed file from: {event.src_path} to: {processed_path}")
                    shutil.move(event.src_path, processed_path)
                    print(f"Processed file moved to: {processed_path}")
            except Exception as e:
                print(f"Error handling file {event.src_path}: {e}")


if __name__ == "__main__":
    path_to_watch = '/Users/puppuccino/PycharmProjects/inner_mon/Charmed'
    print(f"Starting to monitor directory: {path_to_watch}")
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=False)
    observer.start()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    print("Observer stopped.")
