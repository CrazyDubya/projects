import sys
from PyQt6.QtWidgets import QApplication
from main_window import MainWindow

def main():
    """The main function of the application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
