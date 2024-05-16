
# Inner Monologue Manager

## Overview
`inner_monologue.py` is a Python script designed to manage inner monologues for AI assistants. It uses an API wrapper to process user input and generate internal monologues, thoughts, and questions for users. The script includes features for managing long and short-term memory, handling rate limits, and formatting output. Additionally, it provides a GUI for user interaction.

## Features
- **Inner Monologue Management**: Process user input and generate internal monologues, thoughts, and questions.
- **Memory Management**: Handle long-term and short-term memory for improved responses.
- **Rate Limit Handling**: Manage API call rate limits and token usage.
- **Output Formatting**: Format the output with proper markdown and code blocks.
- **GUI Interface**: Provide a graphical interface for user interaction and input.

## Usage
1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install Dependencies**:
    ```bash
    pip install PyQt5
    ```

3. **Run the Script**:
    ```bash
    python inner_monologue.py
    ```

4. **Interact with the GUI**:
   - Enter your input and specify the number of iterations.
   - Set model type, model name, temperature, and max tokens.
   - Click "Run Inner Monologue" to process the input and view the output.

## Example
```bash
$ python inner_monologue.py
```
- A graphical interface appears, allowing you to enter input, set parameters, and view the output.

## Configuration
- **max_iterations**: Set the maximum number of iterations for the inner monologue processing.
- **system_prompt**: Define the system prompt for the AI assistant.
- **max_requests_per_minute**: Set the maximum number of API requests per minute.
- **max_tokens_per_minute**: Set the maximum number of tokens per minute.
- **delay_after_requests**: Set the delay duration after a certain number of requests.
- **delay_duration**: Define the duration of the delay.

## Future Expansion Plans
1. **Enhanced Memory Management**: Improve the handling of long-term and short-term memory.
2. **Advanced Rate Limit Handling**: Implement more sophisticated rate limit management.
3. **Additional Output Formatting Options**: Provide more options for formatting the output.
4. **Improved GUI**: Add more features and customization options to the graphical interface.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License
This project is licensed under the MIT License.
