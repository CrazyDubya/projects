import subprocess

class FfmpegWrapper:
    """A wrapper for the ffmpeg command-line tool."""

    def __init__(self, ffmpeg_path="ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    def run_command(self, command):
        """Runs a given ffmpeg command."""
        try:
            # The command should be a list of strings
            process = subprocess.Popen(
                [self.ffmpeg_path] + command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"ffmpeg command failed: {stderr}")
            return stdout
        except FileNotFoundError:
            raise RuntimeError(
                f"ffmpeg not found at '{self.ffmpeg_path}'. "
                "Please ensure ffmpeg is installed and in your PATH."
            )
        except Exception as e:
            raise RuntimeError(f"An error occurred: {e}")
