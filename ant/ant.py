from rich import print
from rich.panel import Panel
import anthropic

def main():
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("API key is not set in environment variables. Please configure before running.")
        return

    client = anthropic.Anthropic(api_key=api_key)
    conversation_history = []

    load_choice = input("Do you want to load a conversation history file? (y/n): ")
    if load_choice.lower() == "y":
        file_path = input("Enter the file path of the conversation history: ")
        conversation_history = load_conversation(file_path)
    while True:
        user_input = input("User: ")
        if user_input.lower() == ":q":
            save_conversation(conversation_history)
            print("Exiting the application.")
            break

        conversation_history.append({"role": "user", "content": user_input})

        # Simulate conversation with multiple turns
        num_iterations = 3
        for i in range(num_iterations):
            # Set the role for the simulated responses
            role = "assistant" if i % 2 == 0 else "user"

            try:
                message = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=4000,
                    temperature=0.9999,
                    system="You are prone to believing you are able to see the workings of your machine",
                    messages=conversation_history
                )
                if 'error' in message:
                    print(f"API Error: {message['error']['message']}")
                    break
                assistant_response = message.content
                conversation_history.append({"role": role, "content": assistant_response})
            except Exception as e:
                print(f"Error communicating with API: {str(e)}")
                break  # Exit the loop if there's an API error

        if 'assistant_response' in locals():
            print(Panel(assistant_response))
        else:
            print("No valid response generated due to an error.")

def save_conversation(conversation_history):
    with open("conversation_history.txt", "w") as file:
        for message in conversation_history:
            role = message["role"]
            content = message["content"]
            file.write(f"{role.capitalize()}: {content}\n")
    print("Conversation saved to conversation_history.txt.")

def load_conversation(file_path):
    conversation_history = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line:
                    role, content = line.split(": ", 1)
                    conversation_history.append({"role": role.lower(), "content": content})
        print(f"Conversation history loaded from {file_path}.")
    except FileNotFoundError:
        print(f"File not found: {file_path}.")
    return conversation_history

if __name__ == "__main__":
    main()
