from PyQt6.QtWidgets import QMainWindow

class MainWindow(QMainWindow):
    """The main window of the application."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FFmpeg GUI")
        # Set initial size
        self.resize(800, 600)
