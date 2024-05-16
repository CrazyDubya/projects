# chatroom.py
import tkinter as tk
from datetime import datetime
from harmonized_api_wrappers import APIWrapper
from prometheus_client import start_http_server, Summary
import random
import time

# Create a metric to track time spent and requests made.
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')


class SharedWindow:
    @REQUEST_TIME.time()
    def __init__(self, root, title):
        self.window = tk.Frame(root, bg="#ffe668", padx=10, pady=10, borderwidth=1, relief=tk.SOLID)
        self.label = tk.Label(self.window, text=title, font=("Arial", 12, "bold"), bg="#FFFFFF", fg="#333333")
        self.label.grid(row=0, column=0)
        self.task_logs = []

class ChatroomWindow(SharedWindow):
    @REQUEST_TIME.time()
    def __init__(self, root, filename):
        super().__init__(root, "Chatroom")
        self.filename = filename
        self.text_edit = tk.Text(self.window, height=20, width=80, font=("Arial", 14), bg="#F5F5F5", fg="#FF0000", padx=10, pady=10)
        self.text_edit.grid(row=1, column=0, sticky="nsew")
        self.window.grid_rowconfigure(1, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        self.window.after(1000, self.check_file)

    @REQUEST_TIME.time()
    def check_file(self):
        with open(self.filename, "r") as f:
            lines = f.readlines()
        self.text_edit.delete(1.0, tk.END)
        self.text_edit.insert(tk.END, "".join(lines))
        self.window.after(1000, self.check_file)

class ExpChatroomWindow(ChatroomWindow):
    @REQUEST_TIME.time()
    def __init__(self, root, filename):
        super().__init__(root, filename)
        scrollbar = tk.Scrollbar(self.window)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.text_edit.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.text_edit.yview)

    @REQUEST_TIME.time()
    def check_file(self):
        at_bottom = self.text_edit.yview()[1] == 1.0
        super().check_file()
        if at_bottom:
            self.text_edit.see(tk.END)


class InputWindow(SharedWindow):
    @REQUEST_TIME.time()
    def __init__(self, root, username, system_prompt, model_type, model_name, chatroom_window):
        super().__init__(root, f"{username} ({model_name})")
        self.username = username
        self.system_prompt = system_prompt
        self.model_type = model_type
        self.model_name = model_name
        self.chatroom_window = chatroom_window
        self.temperature = 0.7
        self.max_tokens = 500

        self.text_input = tk.Entry(self.window, width=40, font=("Arial", 10))
        self.text_input.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.send_button = tk.Button(self.window, text='Send', command=self.send_message, font=("Arial", 10), bg="#2986cc", fg="#2986cc", activebackground="#45a049", activeforeground="#2986cc", padx=10, pady=5)
        self.send_button.grid(row=1, column=1, padx=5, pady=5)

        self.settings_button = tk.Button(self.window, text='Settings', command=self.open_settings, font=("Arial", 10), bg="#2986cc", fg="#2986cc", activebackground="#45a049", activeforeground="#2986cc", padx=10, pady=5)
        self.settings_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

    @REQUEST_TIME.time()
    def open_settings(self):
        settings_window = tk.Toplevel(self.window)
        settings_window.title(f"{self.username} Settings")

        system_prompt_label = tk.Label(settings_window, text="System Prompt:")
        system_prompt_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        system_prompt_entry = tk.Entry(settings_window, width=50)
        system_prompt_entry.insert(0, self.system_prompt)
        system_prompt_entry.grid(row=0, column=1, padx=5, pady=5)

        temperature_label = tk.Label(settings_window, text="Temperature:")
        temperature_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        temperature_entry = tk.Entry(settings_window, width=10)
        temperature_entry.insert(0, str(self.temperature))
        temperature_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        max_tokens_label = tk.Label(settings_window, text="Max Tokens:")
        max_tokens_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        max_tokens_entry = tk.Entry(settings_window, width=10)
        max_tokens_entry.insert(0, str(self.max_tokens))
        max_tokens_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        @REQUEST_TIME.time()
        def save_settings():
            self.system_prompt = system_prompt_entry.get()
            self.temperature = float(temperature_entry.get())
            self.max_tokens = int(max_tokens_entry.get())
            settings_window.destroy()

        save_button = tk.Button(settings_window, text="Save", command=save_settings)
        save_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    @REQUEST_TIME.time()
    def send_message(self):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        user_input = self.text_input.get()

        api_wrapper = APIWrapper()
        response = api_wrapper.process_model(
            self.model_type, self.model_name,
            system_prompt=self.system_prompt,
            refined_input=user_input,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        with open(self.chatroom_window.filename, "a") as f:
            f.write(f"{timestamp} - {self.username} (User): {user_input}\n")
            f.write(f"{timestamp} - {self.username} (Assistant): {response}\n")

        self.chatroom_window.check_file()
        self.text_input.delete(0, tk.END)
if __name__ == "__main__":
    start_http_server(8000)
    root = tk.Tk()
    root.title("AI Chatroom")

    chatroom_window = ExpChatroomWindow(root, "chat_log.txt")
    chatroom_window.window.grid(row=0, column=0, sticky="nsew")

    input_frame = tk.Frame(root)
    input_frame.grid(row=1, column=0, sticky="nsew")

    for i in range(3):
        input_frame.grid_rowconfigure(i, weight=1)
    for j in range(4):
        input_frame.grid_columnconfigure(j, weight=1)

    input_windows = [
        InputWindow(input_frame, "Claude", "<name>Claude</name><role>Helpful Assistant</role><task>Assist the user with their queries and engage in conversation.</task>", "CLAUDE_MODELS", "haiku", chatroom_window),
        InputWindow(input_frame, "GPT Poe", "<name>GPT Poe</name><role>Poet</role><task>Write creative and thought-provoking poetry based on user prompts.</task>", "OPENAI_MODELS", "gpt-3.5-turbo-0125", chatroom_window),
        InputWindow(input_frame, "CEO-Stockton", "<name>CEO-Stockton</name><role>Business Consultant</role><task>Offer strategic advice and insights on business-related matters.</task>", "LOCAL_MODELS", "mixtral-8x7b-local", chatroom_window),
        InputWindow(input_frame, "Codellama", "<name>Codellama</name><role>Coding Assistant</role><task>Assist with coding tasks and provide programming solutions.</task>", "PERPLEXITY_MODELS", "codellama-70b-instruct", chatroom_window),
        InputWindow(input_frame, "WizardCoder", "<name>WizardCoder</name><role>Coding Wizard</role><task>Provide advanced coding techniques and optimized solutions.</task>", "LOCAL_MODELS", "WizardCoder-17b", chatroom_window),
        InputWindow(input_frame, "Gemini Pro", "<name>Gemini Pro</name><role>Gemini Assistant</role><task>Provide information and engaging conversation using the Gemini model.</task>", "GEMINI_MODELS", "gemini-pro", chatroom_window),
        InputWindow(input_frame, "Haiku", "<name>Moderator</name><role>Conversation Moderator</role><task>Ensure the conversation remains respectful and on-topic.</task>", "CLAUDE_MODELS", "sonnet", chatroom_window),
        InputWindow(input_frame, "Sonnet", "<name>Socrates</name><role>Philosopher</role><task>Engage in philosophical discussions and encourage critical thinking.</task>", "CLAUDE_MODELS", "haiku", chatroom_window)
    ]
    for i, input_window in enumerate(input_windows):
        row = i // 4
        col = i % 4
        input_window.window.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    tk.mainloop()
