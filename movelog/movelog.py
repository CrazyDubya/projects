# movelog.py
import shutil

# Set the directory where you want to search for log files
directory = os.getcwd()  # Gets the current working directory

# Path for the log directory
log_directory = os.path.join(directory, 'log')
txt_directory = os.path.join(directory, 'txt')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
    os.makedirs(txt_directory)
# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.log'):  # Checks if the file is a log file
        # Full path of the file
        file_path = os.path.join(directory, filename)
        # Move the file to the log directory
        shutil.move(file_path, log_directory)
        print(f'Moved: {filename}')


print('All .log files have been moved.')

for filename in os.listdir(directory):
    if filename.endswith('.txt'):  # Checks if the file is a txt file
        # Full path of the file
        file_path = os.path.join(directory, filename)
        # Move the file to the log directory
        shutil.move(file_path, txt_directory)
        print(f'Moved: {filename}')

        print('All .txt files have been moved.')
