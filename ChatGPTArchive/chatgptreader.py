import json
import os


def load_conversation(file_path):
    with open(file_path, 'r') as file:
        conversation_data = json.load(file)
    return conversation_data


def format_conversation(conversation):
    formatted_conversation = f"Conversation ID: {conversation['conversation_id']}\n"
    formatted_conversation += f"Title: {conversation['title']}\n\n"

    for message in conversation['messages']:
        author = message['author']
        content_parts = message['content']

        # Handle content parts which might be dicts or strings
        if content_parts and isinstance(content_parts[0], dict):
            content = "\n".join([part.get('text', '') for part in content_parts])
        else:
            content = "\n".join(content_parts)

        create_time = message['create_time']
        update_time = message['update_time']

        formatted_conversation += f"Author: {author}\n"
        formatted_conversation += f"Content: {content}\n"
        formatted_conversation += f"Create Time: {create_time}\n"
        formatted_conversation += f"Update Time: {update_time}\n"
        formatted_conversation += "-" * 40 + "\n"

    return formatted_conversation


def main():
    # Prompt the user for the path to the gptlogs directory
    gptlogs_dir = input("Enter the path to the gptlogs directory: ").strip()

    # List the JSON files in the gptlogs directory
    json_files = [f for f in os.listdir(gptlogs_dir) if f.endswith('.json')]

    if not json_files:
        print("No JSON files found in the specified directory.")
        return

    # Display the available JSON files
    print("Available JSON files:")
    for idx, json_file in enumerate(json_files):
        print(f"{idx + 1}. {json_file}")

    # Prompt the user to select a JSON file
    file_index = int(input("Enter the number of the JSON file to load: ")) - 1

    if file_index < 0 or file_index >= len(json_files):
        print("Invalid selection.")
        return

    # Load the selected JSON file
    selected_file = json_files[file_index]
    file_path = os.path.join(gptlogs_dir, selected_file)

    conversation = load_conversation(file_path)

    # Format the conversation for readability
    formatted_conversation = format_conversation(conversation)

    # Output the formatted conversation
    print(formatted_conversation)


if __name__ == "__main__":
    main()
