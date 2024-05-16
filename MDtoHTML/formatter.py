# formatter.py
from tkinter import simpledialog
import webbrowser
import markdown2
import tempfile
import os

def show_markdown_in_browser(markdown_content):
    # Convert Markdown to HTML
    html_content = markdown2.markdown(markdown_content)
    html_with_encoding = f'<!DOCTYPE html><html><head><meta charset="UTF-8"></head><body>{html_content}</body></html>'

    # Create a temporary HTML file
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as temp_file:
        temp_file.write(html_content)
        temp_file_path = temp_file.name

    # Open the temporary HTML file in the default web browser
    webbrowser.open('file://' + os.path.realpath(temp_file_path))

def get_user_input():
    # Create a simple dialog to get Markdown content from the user
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    markdown_content = simpledialog.askstring("Input", "Paste your Markdown content here:")
    if markdown_content:
        # Automatically add triple quotes around the input string
        markdown_content_formatted = f'''{markdown_content}'''
        show_markdown_in_browser(markdown_content_formatted)

get_user_input()
