
# Ant

## Overview
`ant.py` is a Python script designed to facilitate conversational interactions using the Anthropic API. It allows users to simulate conversations, save and load conversation histories, and handle multiple turns of dialogue. The script leverages the `rich` library for enhanced console output.

## Features
- **API Key Handling**: Ensures the Anthropic API key is set in the environment variables before running.
- **Conversation History**:
  - Option to load a conversation history from a file.
  - Saves the conversation history to a file upon exiting.
- **Simulated Conversation**:
  - Handles multiple turns of dialogue between the user and the assistant.
  - Uses the Anthropic API to generate responses.
  - Displays the assistant's response using the `rich` library.
- **Error Handling**:
  - Provides informative messages if the API key is missing or if there are errors communicating with the API.
  - Handles file not found errors when loading conversation history.

## Usage
1. **Set Up Environment Variables**:
   - Ensure the Anthropic API key is set in your environment variables:
     ```bash
     export ANTHROPIC_API_KEY=your_api_key_here
     ```

2. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

3. **Run the Script**:
    ```bash
    python ant.py
    ```

4. **Interact with the Script**:
    - Optionally load a conversation history file when prompted.
    - Enter your messages when prompted by `User:`.
    - Type `:q` to exit and save the conversation history.

## Example
```bash
$ python ant.py
Do you want to load a conversation history file? (y/n): y
Enter the file path of the conversation history: previous_conversation.txt
User: Hi there!
Assistant: Hello! How can I help you today?
User: What is the weather like?
Assistant: I'm not sure, but you can check your local weather forecast.
User: :q
Conversation saved to conversation_history.txt.
Exiting the application.
```

## Future Expansion Plans
1. **Enhanced Error Handling**: Improve error messages and provide options for recovery.
2. **Customizable Configurations**: Allow users to configure the model parameters and other settings.
3. **GUI Interface**: Develop a graphical user interface for a more user-friendly experience.
4. **Integration with Other APIs**: Expand the script to integrate with additional APIs for more features.
5. **Conversation Analysis**: Implement features for analyzing and summarizing conversation histories.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License
This project is licensed under the MIT License.
