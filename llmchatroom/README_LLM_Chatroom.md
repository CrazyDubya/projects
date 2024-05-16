
# LLM Chatroom

## Overview
`llmchatroom.py` is a Python script designed to facilitate conversations between different language models (LLMs). The script allows users to specify the models and initiate a conversation based on a given task. The script supports models that require API keys and handles the conversation in multiple turns.

## Features
- **Multi-Model Conversation**: Facilitate conversations between different language models.
- **API Key Support**: Handle models that require API keys for authentication.
- **Conversation Saving**: Save the conversation to a file for later review.
- **Token Usage Tracking**: Track the total token usage during the conversation.

## Usage
1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Run the Script**:
    ```bash
    python llmchatroom.py
    ```

3. **Interact with the Script**:
   - Enter model numbers separated by commas.
   - Enter the initial task/message for the LLMs.
   - Enter the number of turns for the conversation.
   - Provide system prompt bases for each selected model.

## Example
```bash
$ python llmchatroom.py
```
- Follow the prompts to set up and initiate the conversation between the LLMs.

## Configuration
- **llm_configs**: Configure the models with their respective names, base URLs, ports, and API keys (if needed).

## Future Expansion Plans
1. **Enhanced Model Support**: Add support for more models and their respective configurations.
2. **Advanced Conversation Features**: Implement features such as conversation branching and context preservation.
3. **Improved Error Handling**: Enhance error handling for better resilience and robustness.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License
This project is licensed under the MIT License.
