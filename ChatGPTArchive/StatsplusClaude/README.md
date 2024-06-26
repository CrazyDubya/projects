# Conversation Analysis Tool

## Description

This Conversation Analysis Tool is a Python-based application designed to perform deep analysis on conversations between users and AI assistants (such as ChatGPT). It uses Natural Language Processing (NLP) techniques to extract insights from the conversations and leverages Claude, an AI assistant, for further interpretation of the results.

## Features

- Local NLP analysis of conversations, including:
  - Basic analysis: word count, sentence count, top words, parts of speech distribution, sentiment analysis, lexical diversity, and average sentence length.
  - Advanced analysis: topic shifts, co-occurrence analysis, topic modeling, named entity recognition, readability scores, conversation flow, and user-AI interaction metrics.
- Integration with Claude AI for deeper insights and interpretations.
- Ability to save and load analysis results.
- Interactive command-line interface for easy navigation and analysis.
- Support for both basic and advanced analysis modes.
- Extraction and categorization of requests for further analysis.

## Requirements

- Python 3.7+
- Required Python packages (install via `pip install -r requirements.txt`):
  - nltk
  - numpy
  - networkx
  - scikit-learn
  - textstat
  - anthropic

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/conversation-analysis-tool.git
   cd conversation-analysis-tool
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Anthropic API key as an environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Update the `BASE_DIR` variable in the script to point to your directory containing the conversation JSON files.

## Usage

1. Run the script:
   ```
   python conversation_analysis_tool.py
   ```

2. Follow the on-screen prompts to:
   - Analyze a new conversation
   - Load a previous analysis
   - Exit the program

3. When analyzing a new conversation:
   - Choose the conversation file you want to analyze
   - Select the analysis mode (basic or advanced)
   - Review the analysis results and Claude's insights
   - Ask follow-up questions or request further analysis

4. To end the analysis session, type '/bye' when prompted for input.

## File Structure

- `conversation_analysis_tool.py`: The main Python script containing all the analysis functions and the command-line interface.
- `analyzed_conversations/`: Directory containing the conversation JSON files to be analyzed.
- `analysis_results/`: Directory where analysis results are saved.
- `claude_requests.csv`: CSV file containing extracted requests for further analysis.

## Notes

- Ensure that your conversation JSON files are properly formatted with 'user' and 'assistant' roles for each message.
- The advanced analysis mode may take longer to process, especially for large conversations.
- Claude's insights are generated based on the local analysis results and may vary.
- This tool is designed for research and analysis purposes and should be used responsibly.

## Limitations

- The tool currently only supports JSON-formatted conversation files.
- Performance may be affected by the size and complexity of the conversations being analyzed.
- The accuracy of the analysis depends on the quality and structure of the input data.

## Contributing

Contributions to improve the Conversation Analysis Tool are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

[MIT License](LICENSE)
