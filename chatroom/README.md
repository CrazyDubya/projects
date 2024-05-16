
# AI Chatroom

## Overview
`chatroom.py` is a Python script that creates a multi-user AI chatroom using Tkinter for the GUI and various AI models for generating responses. The script integrates Prometheus for tracking metrics and supports dynamic chat updates.

## Features
- **AI Chatroom**: Interactive chatroom with multiple AI models.
- **Prometheus Integration**: Tracks request processing time.
- **Dynamic Updates**: Automatically updates the chat window with new messages.
- **Multiple AI Models**: Supports Claude, GPT, Local models, and Gemini.
- **User Interaction**: Send messages to AI models and receive real-time responses.
- **Settings Management**: Adjust model parameters through a settings interface.

## Usage
1. **Set Up Environment Variables**:
   - Ensure the necessary API keys are set in your environment variables:
     ```bash
     export CLAUDE_API_KEY=your_claude_api_key
     export OPENAI_API_KEY=your_openai_api_key
     export MONSTER_API_KEY=your_monster_api_key
     export PERPLEXITY_API_KEY=your_perplexity_api_key
     export GOOGLE_API_KEY=your_google_api_key
     ```

2. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Script**:
    ```bash
    python chatroom.py
    ```

5. **Interact with the Chatroom**:
   - The chatroom will open with multiple AI models.
   - Send messages to AI models and receive responses in real-time.
   - Adjust settings such as system prompt, temperature, and max tokens as needed.

## Example
```bash
$ python chatroom.py
Starting Prometheus server on port 8000...
Starting AI Chatroom...
```

## Configuration
- **prometheus_client**: Starts a Prometheus HTTP server on port 8000 to track metrics.

## Future Expansion Plans
1. **Enhanced Error Handling**: Improve error messages and provide options for recovery.
2. **Customizable UI**: Allow users to customize the chatroom interface.
3. **Additional AI Models**: Integrate more AI models for greater variety in responses.
4. **Advanced Metrics**: Implement more detailed metrics for tracking performance and usage.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License
This project is licensed under the MIT License.
