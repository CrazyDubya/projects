import csv
import hashlib
import json
import os
import re
import string
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple

import networkx as nx
import nltk
import numpy as np
from anthropic import Anthropic
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

BASE_DIR = 'WHERE/CHATGPT/JSON?STORED'  # Update this path
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis_results')  # Directory to store analysis results

# Create analysis directory if it doesn't exist
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def preprocess_text(text: str) -> List[str]:
    """Preprocess text by tokenizing, lowercasing, and removing punctuation and stopwords."""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words and token not in string.punctuation]


def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities from the text."""
    chunks = nltk.ne_chunk(nltk.pos_tag(word_tokenize(text)))
    entities = defaultdict(list)
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            entities[chunk.label()].append(' '.join(c[0] for c in chunk))
    return dict(entities)


def calculate_readability_scores(text: str) -> Dict[str, float]:
    """Calculate readability scores for the text."""
    return {
        "flesch_reading_ease": flesch_reading_ease(text),
        "flesch_kincaid_grade": flesch_kincaid_grade(text)
    }


def analyze_conversation_flow(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the flow of the conversation."""
    turn_lengths = [len(word_tokenize(msg['content'])) for msg in messages]
    return {
        "avg_turn_length": np.mean(turn_lengths),
        "turn_length_variance": np.var(turn_lengths),
        "conversation_length": len(messages)
    }


def analyze_user_chatgpt_interaction(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the interaction between the user and ChatGPT."""
    user_messages = []
    chatgpt_messages = []

    for msg in messages:
        if isinstance(msg, dict):
            if 'role' in msg:
                if msg['role'] == 'user':
                    user_messages.append(msg)
                elif msg['role'] == 'assistant':
                    chatgpt_messages.append(msg)
            elif 'user' in msg:
                user_messages.append(msg)
            elif 'ChatGPT' in msg:
                chatgpt_messages.append(msg)
        else:
            print(f"Unexpected message format: {msg}")

    if not user_messages and not chatgpt_messages:
        print("Warning: No user or ChatGPT messages found. Check the conversation structure.")
        return {
            "error": "Unable to analyze user-ChatGPT interaction due to unexpected conversation structure."
        }

    total_messages = len(user_messages) + len(chatgpt_messages)

    return {
        "user_message_count": len(user_messages),
        "chatgpt_message_count": len(chatgpt_messages),
        "user_message_ratio": len(user_messages) / total_messages if total_messages > 0 else 0,
        "avg_user_message_length": np.mean(
            [len(word_tokenize(msg.get('content', ''))) for msg in user_messages]) if user_messages else 0,
        "avg_chatgpt_message_length": np.mean(
            [len(word_tokenize(msg.get('content', ''))) for msg in chatgpt_messages]) if chatgpt_messages else 0,
        "user_question_count": sum(1 for msg in user_messages if msg.get('content', '').strip().endswith('?')),
        "chatgpt_code_block_count": sum(1 for msg in chatgpt_messages if '```' in msg.get('content', ''))
    }


def analyze_topic_shifts(messages: List[Dict[str, Any]]) -> List[Tuple[int, List[str]]]:
    """Conduct a chronological analysis of topic shifts."""
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    topic_shifts = []

    for i, msg in enumerate(messages):
        content = msg.get('content', '')
        if not content.strip():  # Skip empty messages
            continue
        try:
            tfidf_matrix = vectorizer.fit_transform([content])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            top_topics = [feature_names[i] for i in tfidf_scores.argsort()[::-1][:5]]
            topic_shifts.append((i, top_topics))
        except ValueError:
            # If vocabulary is empty, skip this message
            continue

    return topic_shifts


def perform_co_occurrence_analysis(text: str, window_size: int = 5) -> nx.Graph:
    """Perform co-occurrence analysis on key terms."""
    words = preprocess_text(text)
    co_occurrence = nx.Graph()

    if len(words) < 2:
        return co_occurrence  # Return empty graph if not enough words

    for i in range(len(words)):
        for j in range(i + 1, min(i + window_size, len(words))):
            if co_occurrence.has_edge(words[i], words[j]):
                co_occurrence[words[i]][words[j]]['weight'] += 1
            else:
                co_occurrence.add_edge(words[i], words[j], weight=1)

    return co_occurrence


def topic_modeling(texts: List[str], num_topics: int = 5) -> Tuple[List[List[Tuple[str, float]]], np.ndarray]:
    """Use LDA for topic modeling."""
    if not texts:
        return [], np.array([])

    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # If vocabulary is empty, return empty results
        return [], np.array([])

    if tfidf_matrix.shape[1] == 0:
        return [], np.array([])

    lda = LatentDirichletAllocation(n_components=min(num_topics, tfidf_matrix.shape[1]), random_state=42)
    lda.fit(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-10 - 1:-1]
        top_features = [(feature_names[i], topic[i]) for i in top_features_ind]
        topics.append(top_features)

    topic_distribution = lda.transform(tfidf_matrix)
    return topics, topic_distribution


def advanced_sentiment_analysis(text: str) -> Dict[str, float]:
    """Perform advanced sentiment analysis on the text."""
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    # Additional analysis for frustration detection
    frustration_keywords = ['error', 'bug', 'problem', 'issue', 'fail', 'crash', 'doesn\'t work', 'broken']
    frustration_score = sum(text.lower().count(keyword) for keyword in frustration_keywords) / len(text.split())

    sentiment_scores['frustration'] = frustration_score
    return sentiment_scores


def analyze_conversation_locally(conversation: Dict, mode: str = 'basic') -> Dict[str, Any]:
    """Perform comprehensive local NLP analysis on the conversation."""
    try:
        messages = conversation.get('messages', [])
        if not messages:
            return {"error": "No messages found in the conversation."}

        full_content = " ".join([msg.get('content', '') for msg in messages if isinstance(msg, dict)])

        if not full_content.strip():
            return {"error": "The conversation is empty or contains only whitespace."}

        words = preprocess_text(full_content)
        sentences = sent_tokenize(full_content)

        if not words:
            return {"error": "No meaningful words found in the conversation after preprocessing."}

        basic_analysis = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "top_words": FreqDist(words).most_common(10),
            "pos_distribution": dict(Counter(tag for word, tag in nltk.pos_tag(words))),
            "sentiment_analysis": advanced_sentiment_analysis(full_content),
            "lexical_diversity": len(set(words)) / len(words) if words else 0,
            "avg_sentence_length": np.mean([len(word_tokenize(sentence)) for sentence in sentences]) if sentences else 0
        }

        if mode == 'advanced':
            advanced_analysis = {
                "topic_shifts": analyze_topic_shifts(messages),
                "co_occurrence_graph": list(perform_co_occurrence_analysis(full_content).edges(data=True)),
                "topics": topic_modeling([msg.get('content', '') for msg in messages if isinstance(msg, dict)])[0],
                "syntactic_complexity": len(words) / len(sentences) if sentences else None,
                "named_entities": extract_named_entities(full_content),
                "readability_scores": calculate_readability_scores(full_content),
                "conversation_flow": analyze_conversation_flow(messages),
                "user_chatgpt_interaction": analyze_user_chatgpt_interaction(messages)
            }
            return {**basic_analysis, **advanced_analysis}

        return basic_analysis
    except Exception as e:
        print(f"Error in analyze_conversation_locally: {str(e)}")
        return {"error": f"An error occurred during analysis: {str(e)}"}


def get_file_hash(file_path: str) -> str:
    """Generate a hash for the file content."""
    with open(file_path, 'rb') as file:
        return hashlib.md5(file.read()).hexdigest()


def save_analysis_results(file_path: str, local_analysis: Dict, claude_conversation: List[Dict], mode: str):
    """Save both local and Claude's analysis results."""
    file_hash = get_file_hash(file_path)
    analysis_file = os.path.join(ANALYSIS_DIR, f"{file_hash}_analysis.json")

    results = {
        "original_file": os.path.basename(file_path),
        "mode": mode,
        "local_analysis": local_analysis,
        "claude_conversation": claude_conversation
    }

    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


def load_analysis_results(file_path: str) -> Dict:
    """Load existing analysis results if available."""
    file_hash = get_file_hash(file_path)
    analysis_file = os.path.join(ANALYSIS_DIR, f"{file_hash}_analysis.json")

    if os.path.exists(analysis_file):
        with open(analysis_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def extract_and_save_requests(response: str, file_path: str) -> None:
    """Extract requests for further analysis from Claude's response and save them to a CSV file."""
    requests = re.findall(r'<requested4analysis>(.*?)</requested4analysis>', response, re.DOTALL)

    if not requests:
        print("No requests for additional analysis found.")
        return

    csv_file = os.path.join(ANALYSIS_DIR, 'claude_requests.csv')
    file_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Conversation', 'Request', 'Category', 'Priority'])

        for req in requests:
            category, priority = categorize_request(req)
            writer.writerow([os.path.basename(file_path), req.strip(), category, priority])

    print(f"Saved {len(requests)} requests for further analysis.")


def categorize_request(request: str) -> Tuple[str, str]:
    """Categorize the request and assign a priority."""
    categories = {
        'statistical': ['correlation', 'regression', 'distribution', 'hypothesis', 'test', 'variance'],
        'linguistic': ['syntax', 'semantic', 'discourse', 'pragmatic', 'lexical'],
        'semantic': ['topic', 'concept', 'entity', 'relation', 'ontology'],
        'interaction': ['turn-taking', 'response time', 'engagement', 'clarification'],
        'technical': ['code', 'algorithm', 'framework', 'library', 'api']
    }

    request_lower = request.lower()
    for category, keywords in categories.items():
        if any(keyword in request_lower for keyword in keywords):
            return category, 'high'

    # If no specific category is found, use 'general' with medium priority
    return 'general', 'medium'


def format_analysis_summary(analysis: Dict, mode: str) -> str:
    """Format the analysis summary for Claude's input."""
    summary = f"""
    Analysis Mode: {mode}
    Word Count: {analysis['word_count']}
    Sentence Count: {analysis['sentence_count']}
    Top Words: {', '.join([f"{word}({count})" for word, count in analysis['top_words']])}
    Parts of Speech Distribution: {analysis['pos_distribution']}
    Lexical Diversity: {analysis['lexical_diversity']:.2f}
    Average Sentence Length: {analysis['avg_sentence_length']:.2f}
    Sentiment Analysis:
    {json.dumps(analysis['sentiment_analysis'], indent=2)}
    """

    if mode == 'advanced':
        summary += f"""
    Top 3 Topics:
    {json.dumps(analysis['topics'][:3], indent=2)}
    Top 10 Co-occurrences:
    {json.dumps(analysis['co_occurrence_graph'][:10], indent=2)}
    Named Entities:
    {json.dumps(analysis['named_entities'], indent=2)}
    Readability Scores:
    {json.dumps(analysis['readability_scores'], indent=2)}
    Conversation Flow:
    {json.dumps(analysis['conversation_flow'], indent=2)}
    User-ChatGPT Interaction:
    {json.dumps(analysis['user_chatgpt_interaction'], indent=2)}
    """

    return summary


def analyze_conversation(file_path: str, conversation: Dict, mode: str = 'basic') -> None:
    """Perform deep analysis of the conversation using local NLP and Claude."""
    existing_analysis = load_analysis_results(file_path)

    if existing_analysis:
        if 'mode' in existing_analysis and existing_analysis['mode'] == mode:
            print("Loading existing analysis results...")
            local_analysis = existing_analysis.get('local_analysis', {})
            claude_conversation = existing_analysis.get('claude_conversation', [])
        else:
            print(f"Existing analysis found, but mode doesn't match. Performing new {mode} analysis...")
            local_analysis = analyze_conversation_locally(conversation, mode)
            claude_conversation = []
    else:
        print(f"Performing new {mode} analysis...")
        local_analysis = analyze_conversation_locally(conversation, mode)
        claude_conversation = []

    if "error" in local_analysis:
        print(f"Error in local analysis: {local_analysis['error']}")
        return

    if not claude_conversation:
        analysis_summary = format_analysis_summary(local_analysis, mode)

        initial_prompt = f"""
        Based on the following {mode} NLP analysis of a technical conversation between a user and ChatGPT, please provide deep insights and interpretations:

        {analysis_summary}

        Please consider:
        1. How do the topic shifts and conversation flow reflect the structure of the technical discussion?
        2. What insights can we draw from the co-occurrence analysis and named entities about the relationships between technical concepts?
        3. How does the sentiment analysis and user-ChatGPT interaction metrics reflect the participants' experiences with the technical content?
        4. What hidden themes or concepts are revealed by the topic modeling that might not be immediately apparent?
        5. How do the readability scores, syntactic complexity, and lexical diversity reflect the level of expertise in the conversation?
        6. Based on this analysis, what can we infer about the nature, depth, and context of this technical discussion?
        7. Are there any patterns or anomalies in the data that warrant further investigation?
        8. How does the conversation demonstrate the capabilities and limitations of ChatGPT in handling technical discussions?
        9. What recommendations would you make for improving the quality of such technical conversations between users and AI assistants?

        Provide a detailed interpretation of these results, highlighting key insights and potential areas for further analysis. Feel free to use <think></think> tags to simulate your thought process.

        """
        claude_conversation.append({"role": "user", "content": initial_prompt})

    while True:
        if claude_conversation:
            try:
                message = anthropic.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=4000,
                    messages=claude_conversation
                )
                response = message.content[0].text
                claude_conversation.append({"role": "assistant", "content": response})
                print("\nClaude's Analysis:")
                print(response)

                # Extract and save requests for further analysis
                extract_and_save_requests(response, file_path)
            except Exception as e:
                print(f"Error in getting Claude's response: {e}")
                break

        user_input = input("\nEnter your next question (or '/bye' to return to the main menu): ")
        if user_input.lower() == '/bye':
            print("Returning to the main menu...")
            break

        claude_conversation.append({"role": "user", "content": user_input})
        save_analysis_results(file_path, local_analysis, claude_conversation, mode)

    # Final save before returning to the main menu
    save_analysis_results(file_path, local_analysis, claude_conversation, mode)


def load_conversations(directory: str) -> Dict[str, str]:
    """Load conversations from JSON files in the specified directory."""
    conversations = {}
    for i, filename in enumerate(os.listdir(directory), 1):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            conversations[str(i)] = file_path
    return conversations


def load_conversation_content(file_path: str) -> Dict:
    """Load the content of a single conversation from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def list_saved_analyses():
    """List all saved analyses."""
    analyses = []
    for filename in os.listdir(ANALYSIS_DIR):
        if filename.endswith('_analysis.json'):
            with open(os.path.join(ANALYSIS_DIR, filename), 'r') as f:
                data = json.load(f)
                analyses.append(data['original_file'])
    return analyses


def main():
    while True:
        print("\n1. Analyze a new conversation")
        print("2. Load a previous analysis")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            conversations = load_conversations(BASE_DIR)
            print("\nAvailable conversations:")
            for num, file_path in conversations.items():
                print(f"{num}: {os.path.basename(file_path)}")

            conv_choice = input("\nEnter the number of the conversation you want to analyze: ")
            if conv_choice in conversations:
                file_path = conversations[conv_choice]
                conversation = load_conversation_content(file_path)

                analysis_mode = input("Choose analysis mode (basic/advanced): ").lower()
                if analysis_mode not in ['basic', 'advanced']:
                    print("Invalid mode. Defaulting to basic analysis.")
                    analysis_mode = 'basic'

                analyze_conversation(file_path, conversation, analysis_mode)
            else:
                print("Invalid choice. Please try again.")

        elif choice == '2':
            saved_analyses = list_saved_analyses()
            if not saved_analyses:
                print("No saved analyses found.")
                continue

            print("\nSaved analyses:")
            for i, filename in enumerate(saved_analyses, 1):
                print(f"{i}: {filename}")

            analysis_choice = input("\nEnter the number of the analysis you want to load: ")
            try:
                file_to_load = saved_analyses[int(analysis_choice) - 1]
                file_path = os.path.join(BASE_DIR, file_to_load)
                conversation = load_conversation_content(file_path)
                existing_analysis = load_analysis_results(file_path)
                mode = existing_analysis.get('mode', 'basic') if existing_analysis else 'basic'
                analyze_conversation(file_path, conversation, mode)
            except (ValueError, IndexError):
                print("Invalid choice. Please try again.")

        elif choice == '3':
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
