
# Chatter User

## Overview
`chatter-user.py` is a Python script that facilitates conversations between two selected Large Language Models (LLMs). The script allows users to configure and select models, send requests, and log the conversations.

## Features
- **Configuration**: Supports multiple LLMs with specific configurations, including API keys for external models.
- **Dynamic Model Selection**: Users can select two models for interaction.
- **User Interaction**: Prompts for task input, system prompts, and number of turns for the conversation.
- **API Requests**: Sends requests to the specified LLM endpoints and processes responses.
- **Conversation Logging**: Saves the conversation to a timestamped file.
- **Validation**: Ensures valid inputs for the number of turns and model selection.

## Usage
1. **Set Up Environment Variables**:
   - Ensure the necessary API keys are set in your environment variables:
     ```bash
     export EXTERNALMODEL_API_KEY=your_externalmodel_api_key
     ```

2. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

3. **Run the Script**:
    ```bash
    python chatter-user.py
    ```

4. **Interact with the Script**:
   - Follow the prompts to select models, enter the task, system prompts, and number of turns.
   - The script will facilitate the conversation between the selected models and log the conversation to a file.

## Example
```bash
$ python chatter-user.py
Available models: 1. Mixtral, 2. Gwen, 3. ExternalModel
Enter two model numbers separated by a comma (e.g., 1,2): 1,3
Enter the initial task/message for the LLMs: Discuss the impact of AI on society
Enter number of turns (2-100, default 10): 5
Enter system prompt base for Model 1: You are Mixtral. Discuss the topic with {other_model}.
Enter system prompt base for Model 3: You are ExternalModel. Discuss the topic with {other_model}.
Mixtral: AI has the potential to greatly impact society in various ways...
ExternalModel: Indeed, AI could revolutionize industries and improve efficiency...
...
Conversation saved to Discuss_the_imp_20240515_121045.txt
Token usage: {'total_tokens': 1200}
```

## Configuration
- **llm_configs**: Configuration dictionary for the LLMs, including names, base URLs, ports, and API keys.

## Future Expansion Plans
1. **Enhanced Error Handling**: Improve error messages and provide options for recovery.
2. **Customizable Conversations**: Allow users to customize more conversation parameters.
3. **Additional Models**: Integrate more LLMs for greater variety in interactions.
4. **Advanced Metrics**: Implement detailed metrics for tracking performance and usage.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License
This project is licensed under the MIT License.
