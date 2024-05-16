# _scriptrunner.py


def run_script(script_name):
    return subprocess.Popen(['python', script_name])


if __name__ == '__main__':
    # Names of the scripts to run
    scripts = ['mover.py', 'WatchCharm.py', 'autopy.py']

    # Start all scripts
    processes = [run_script(script) for script in scripts]

    # Wait for all scripts to complete
    for process in processes:
        process.wait()
