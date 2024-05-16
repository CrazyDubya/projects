#jsonreader.py
import json

# Specify the file name
file_name = '../data/conversations.json'

# Open the file and parse the JSON data
with open(file_name, 'r') as file:
    data = json.load(file)

# Iterate through the conversations
for conversation in data:
    # Extract conversation-level information
    title = conversation.get('title')
    create_time = conversation.get('create_time')
    update_time = conversation.get('update_time')
    conversation_id = conversation.get('conversation_id')

    # Print conversation-level information
    print(f"Conversation ID: {conversation_id}")
    print(f"Title: {title}")
    print(f"Create Time: {create_time}")
    print(f"Update Time: {update_time}")
    print("Messages:")

    # Iterate through all the nodes in the conversation mapping
    for node_id, node in conversation['mapping'].items():
        message = node.get('message', {})
        if message:
            message_id = message.get('id')
            author_role = message.get('author', {}).get('role')
            create_time = message.get('create_time')
            content_parts = message.get('content', {}).get('parts', [])
            status = message.get('status')

            # Convert content_parts to a list of strings
            content_parts_str = [str(part) for part in content_parts]

            # Print message-level information
            print(f"  Message ID: {message_id}")
            print(f"  Author Role: {author_role}")
            print(f"  Create Time: {create_time}")
            print(f"  Content: {' '.join(content_parts_str)}")
            print(f"  Status: {status}")

    print("---")
