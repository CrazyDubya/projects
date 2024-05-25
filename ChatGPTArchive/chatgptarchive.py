# chatgptarchive.py
# Description: Parse and save conversations from a JSON file to individual files
import json
import os
import re
from datetime import datetime


def parse_conversations(conversations):
    parsed_data = []

    for conversation in conversations:
        conversation_id = conversation.get('id')
        title = conversation.get('title')

        # Handle None or empty titles
        if not title:
            title = 'conversation'

        title = title.replace(" ", "_")
        messages = conversation.get('mapping', {}).values()

        conversation_data = {
            'conversation_id': conversation_id,
            'title': title,
            'messages': []
        }

        for message_entry in messages:
            if not message_entry or not message_entry.get('message'):
                continue

            message = message_entry['message']
            author = message.get('author', {}).get('role') if message.get('author') else None
            content = message.get('content', {}).get('parts', []) if message.get('content') else []
            create_time = message.get('create_time')
            update_time = message.get('update_time')
            parent_id = message_entry.get('parent')
            children_ids = message_entry.get('children', [])

            conversation_data['messages'].append({
                'message_id': message_entry.get('id'),
                'author': author,
                'content': content,
                'create_time': create_time,
                'update_time': update_time,
                'parent_id': parent_id,
                'children_ids': children_ids
            })

        parsed_data.append(conversation_data)

    return parsed_data


def sanitize_filename(filename):
    """Sanitize the filename by removing invalid characters and ensuring it is not empty."""
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', filename)
    return sanitized if sanitized else 'conversation'


def save_conversations_to_files(parsed_conversations, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename_counts = {}

    for conversation in parsed_conversations:
        title = conversation['title']
        base_filename = sanitize_filename(title[:10])

        if base_filename in filename_counts:
            filename_counts[base_filename] += 1
            filename = f"{base_filename}_{filename_counts[base_filename]}.json"
        else:
            filename_counts[base_filename] = 1
            filename = f"{base_filename}.json"

        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as file:
            json.dump(conversation, file, indent=4)


def main():
    # Prompt the user for the path to the conversations.json file
    conversations_json_path = input("Enter the path to the conversations.json file: ").strip()

    # Load the JSON data
    with open(conversations_json_path, 'r') as file:
        conversations_data = json.load(file)

    # Parse the conversations data
    parsed_conversations = parse_conversations(conversations_data)

    # Define the output directory with a timestamp
    timestamp = datetime.now().strftime("%d%m%y")
    output_dir = os.path.join(os.path.dirname(conversations_json_path), f'gptlogs-{timestamp}')

    # Save each parsed conversation to its own file
    save_conversations_to_files(parsed_conversations, output_dir)

    print(f"Parsed conversations have been saved to {output_dir}")


if __name__ == "__main__":
    main()
