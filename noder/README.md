
# HiveMind Node

## Overview
`Noder.py` is a Python script that facilitates communication with an AI model using XML structured prompts. The script initializes a HiveMind node and manages a conversation state, saving it to a file for continuity between runs. The script is designed to operate within a HiveMind framework, where it can send and receive messages, extract XML responses, and update the conversation state accordingly.

## Features
- **Initialize HiveMind Node**: Set up a solitary HiveMind node with reflective capabilities.
- **Send and Receive Messages**: Communicate with an AI model using structured XML messages.
- **Extract XML Responses**: Parse the XML response from the AI model to extract prompts and code instructions.
- **Save and Load Conversation State**: Persist conversation state to a file for continuity between script executions.
- **Configurable Prompts**: Allows for initial prompts and code prompts to be customized by the user.

## Usage
1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Configure API Key**:
   - Set the API key for the `anthropic` client in the script:
     ```python
     client = anthropic.Anthropic(api_key="your_api_key_here")
     ```

3. **Run the Script**:
    ```bash
    python Noder.py
    ```

## Example
```bash
$ python Noder.py
```
- The script will initialize the HiveMind node, communicate with the AI model, and manage the conversation state.

## Configuration
- **System Prompt**: The initial system prompt used to set up the HiveMind rules.
- **User Prompts and Responses**: The user prompts and responses that guide the conversation with the AI model.
- **Code Prompt**: The initial code prompt can be provided by the user for context.

## Future Expansion Plans
1. **Enhanced Error Handling**: Improve error handling for various failure scenarios.
2. **Additional Node Types**: Support multiple node types with different roles and tasks.
3. **Extended Conversation History**: Maintain a longer conversation history with more detailed state management.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License
This project is licensed under the MIT License.
