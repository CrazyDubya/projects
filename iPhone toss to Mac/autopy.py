# autopy.py
import subprocess
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith('.py'):
            print(f"Detected new file: {event.src_path}, running...")

            # Run the Python script and capture the console output in real-time
            process = subprocess.Popen(['python', event.src_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       universal_newlines=True)

            # Create the console log file
            console_log_path = '/Users/puppuccino/PycharmProjects/inner_mon/Charmed//console30/console_log.txt'
            os.makedirs(os.path.dirname(console_log_path), exist_ok=True)

            # Read and print the console output in real-time
            with open(console_log_path, 'w') as f:
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)

            # Wait for 3 seconds
            time.sleep(3)

            # Capture the desktop image
            desktop_image_path = '/Users/puppuccino/PycharmProjects/inner_mon/Charmed//pyimage/desktop_image.png'
            os.makedirs(os.path.dirname(desktop_image_path), exist_ok=True)
            subprocess.run(['screencapture', '-x', desktop_image_path], check=True)
            print(f"Captured desktop image: {desktop_image_path}")

            # Wait for the Python script to finish
            process.wait()

            print(f"Saved console log: {console_log_path}")


if __name__ == "__main__":
    path = '/Users/puppuccino/PycharmProjects/inner_mon/Charmed/charm'
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
