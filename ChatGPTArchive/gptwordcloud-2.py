import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud

# Ensure the stopwords are downloaded
nltk.download('stopwords')

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)

# Extract messages from the 'mapping' key with a check for NoneType
messages = []
for conversation in conversations_data:
    for node in conversation['mapping'].values():
        if node.get('message') and node['message'].get('content'):
            message_content = node['message']['content'].get('parts', [])
            messages.extend(message_content)

# Filter out non-text messages
text_messages = [message for message in messages if isinstance(message, str)]

# Join all messages into a single string
all_text = ' '.join(text_messages)

# Tokenize the text by words, converting to lowercase
words = re.findall(r'\b\w+\b', all_text.lower())

# Get the list of English stopwords
stop_words = set(stopwords.words('english'))

# Filter out stop words
filtered_words = [word for word in words if word not in stop_words]

# Count the frequency of each word
word_counts = Counter(filtered_words)

# Get the 20 most common words
common_words = word_counts.most_common(1000)

# Display the most common words
print("Most common words:")
for word, count in common_words:
    print(f"{word}: {count}")

# Create and display a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Perform sentiment analysis on each text message
sentiments = [TextBlob(message).sentiment.polarity for message in text_messages if message.strip()]

# Calculate the overall sentiment
overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
print(f"Overall sentiment: {overall_sentiment}")

# Save the results to a file
with open('analysis_results.txt', 'w') as result_file:
    result_file.write("Most common words:\n")
    for word, count in common_words:
        result_file.write(f"{word}: {count}\n")
    result_file.write(f"\nOverall sentiment: {overall_sentiment}\n")
