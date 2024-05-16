#jsonreader2.py
import json
import textwrap

# Specify the input file name
input_file_name = '../data/conversations.json'

# Specify the output file name
output_file_name = 'xxx.txt'

# Open the input file and parse the JSON data
with open(input_file_name, 'r') as input_file:
    data = json.load(input_file)

# Open the output file for writing
with open(output_file_name, 'w') as output_file:
    # Iterate through the conversations
    for conversation in data:
        # Extract conversation-level information
        title = conversation.get('title', 'N/A')
        create_time = conversation.get('create_time', 'N/A')
        update_time = conversation.get('update_time', 'N/A')
        conversation_id = conversation.get('conversation_id', 'N/A')

        # Write conversation-level information to the output file
        output_file.write(f"Conversation ID: {conversation_id}\n")
        output_file.write(f"Title: {title}\n")
        output_file.write(f"Create Time: {create_time}\n")
        output_file.write(f"Update Time: {update_time}\n")
        output_file.write("Messages:\n")

        # Iterate through all the nodes in the conversation mapping
        for node_id, node in conversation['mapping'].items():
            message = node.get('message')
            if message:
                message_id = message.get('id', 'N/A')
                author_role = message.get('author', {}).get('role', 'N/A')
                create_time = message.get('create_time', 'N/A')
                content = message.get('content', {})
                content_parts = content.get('parts', [])
                status = message.get('status', 'N/A')

                # Convert content_parts to a single string
                content_text = ' '.join(str(part) for part in content_parts)

                # Wrap the content text for better readability
                wrapped_content = textwrap.fill(content_text, width=80)

                # Write message-level information to the output file
                output_file.write(f"Message ID: {message_id}\n")
                output_file.write(f"Author Role: {author_role}\n")
                output_file.write(f"Create Time: {create_time}\n")
                output_file.write(f"Content:\n{wrapped_content}\n")
                output_file.write(f"Status: {status}\n\n")

        output_file.write("---\n\n")

print(f"Extracted information saved to: {output_file_name}")
