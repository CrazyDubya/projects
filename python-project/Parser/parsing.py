#!/usr/bin/env python3
"""
✨ CogniSphere: Your Intelligent Knowledge Navigator ✨

Embark on a journey with **CogniSphere**, a sophisticated script that masterfully orchestrates data parsing,
categorization, summarization, knowledge graph creation, natural language querying, real-time
updates, visualization, and persistent storage. Seamlessly handling multiple data formats
(CSV, JSON, text, APIs), CogniSphere crafts a dynamic and intelligent knowledge base that evolves with your
interactions.

👨‍💻 **Author**: OpenAI ChatGPT
📅 **Date**: 2024-09-18

🚀 **Dependencies**:
    - Python 3.8+
    - requests==2.31.0
    - networkx==3.1
    - matplotlib==3.7.2
    - watchdog==3.0.0
    - nltk==3.8.1
    - sqlalchemy==2.0.14
    - Flask==2.3.2
    - Flask-HTTPAuth==4.7.0
    - scikit-learn==1.2.2

🔧 **Installation**:
    ```bash
    pip install requests==2.31.0 networkx==3.1 matplotlib==3.7.2 watchdog==3.0.0 nltk==3.8.1 sqlalchemy==2.0.14 Flask==2.3.2 Flask-HTTPAuth==4.7.0 scikit-learn==1.2.2
    ```

📦 **Initial Setup**:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```
"""

import csv
import json
import logging
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import requests
from flask import Flask, jsonify, request, send_file
from flask_httpauth import HTTPBasicAuth
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import (Column, Integer, String, Text, create_engine, exists,
                        select)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configure logging with a touch of flair
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("cognisphere.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define supported data formats
SUPPORTED_FORMATS = ['csv', 'json', 'text', 'api']

# Initialize SQLAlchemy
Base = declarative_base()
DATABASE_URL = "sqlite:///cognisphere.db"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


class KnowledgeEntry(Base):
    """
    📚 SQLAlchemy model for knowledge base entries.
    """
    __tablename__ = 'knowledge_entries'

    id = Column(Integer, primary_key=True, index=True)
    entry_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, index=True)
    category = Column(String, index=True)
    description = Column(Text)


class User(Base):
    """
    🔐 SQLAlchemy model for users.
    """
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)


# Create all tables
Base.metadata.create_all(bind=engine)


# =========================
# Data Parser Implementations
# =========================

class DataParser(ABC):
    """
    🛠️ Abstract base class for data parsers.
    """

    @abstractmethod
    def parse(self, source: Any) -> List[Dict[str, Any]]:
        pass


class CSVParser(DataParser):
    """
    📄 Parser for CSV files.
    """

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        logger.info(f"🔍 Parsing CSV file: {file_path}")
        data = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
            logger.info(f"✅ Successfully parsed CSV file: {file_path}")
        except FileNotFoundError:
            logger.error(f"❌ CSV file not found: {file_path}")
        except Exception as e:
            logger.error(f"❌ Error parsing CSV file {file_path}: {e}")
        return data


class JSONParser(DataParser):
    """
    📄 Parser for JSON files.
    """

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        logger.info(f"🔍 Parsing JSON file: {file_path}")
        data = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as jsonfile:
                json_data = json.load(jsonfile)
                if isinstance(json_data, dict):
                    data.append(json_data)
                elif isinstance(json_data, list):
                    data.extend(json_data)
            logger.info(f"✅ Successfully parsed JSON file: {file_path}")
        except FileNotFoundError:
            logger.error(f"❌ JSON file not found: {file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error in file {file_path}: {e}")
        except Exception as e:
            logger.error(f"❌ Error parsing JSON file {file_path}: {e}")
        return data


class TextParser(DataParser):
    """
    📄 Parser for plain text files.
    """

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        logger.info(f"🔍 Parsing text file: {file_path}")
        data = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as txtfile:
                content = txtfile.read()
                data.append({"content": content})
            logger.info(f"✅ Successfully parsed text file: {file_path}")
        except FileNotFoundError:
            logger.error(f"❌ Text file not found: {file_path}")
        except Exception as e:
            logger.error(f"❌ Error parsing text file {file_path}: {e}")
        return data


class APIParser(DataParser):
    """
    🌐 Parser for API inputs.
    """

    def parse(self, api_url: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        logger.info(f"🔍 Fetching data from API: {api_url}")
        data = []
        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            json_data = response.json()
            if isinstance(json_data, dict):
                data.append(json_data)
            elif isinstance(json_data, list):
                data.extend(json_data)
            logger.info(f"✅ Successfully fetched data from API: {api_url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API request failed for {api_url}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error from API {api_url}: {e}")
        except Exception as e:
            logger.error(f"❌ Error fetching data from API {api_url}: {e}")
        return data


# =========================
# Parser Factory Function
# =========================

def get_parser(format_type: str) -> Optional[DataParser]:
    """
    🏭 Factory function to get the appropriate parser based on format type.
    """
    parser = None
    if format_type == 'csv':
        parser = CSVParser()
    elif format_type == 'json':
        parser = JSONParser()
    elif format_type == 'text':
        parser = TextParser()
    elif format_type == 'api':
        parser = APIParser()
    else:
        logger.warning(f"⚠️ Unsupported format type: {format_type}")
    return parser


# =========================
# Data Processor Implementation
# =========================

class DataProcessor:
    """
    🧠 Processes and manages data from various sources.
    """

    def __init__(self):
        self.knowledge_base = defaultdict(list)
        self.categories = set()
        self.knowledge_graph = nx.Graph()
        self.lock = threading.Lock()
        self.db_session = SessionLocal()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.initialize_users()
        self.initialize_vectorizer()

    def initialize_users(self):
        """
        🔐 Initialize default users.
        """
        logger.info("🔧 Initializing users")
        try:
            if not self.db_session.query(exists().where(User.username == 'admin')).scalar():
                admin = User(
                    username='admin',
                    password_hash=generate_password_hash('adminpass')
                )
                self.db_session.add(admin)
                self.db_session.commit()
                logger.info("✅ Created default admin user.")
        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"❌ Database error during user initialization: {e}")
        except Exception as e:
            logger.error(f"❌ Error initializing users: {e}")

    def initialize_vectorizer(self):
        """
        📊 Initialize the TF-IDF vectorizer.
        """
        logger.info("📊 Initializing TF-IDF vectorizer")
        try:
            if self.knowledge_base:
                documents = [self._combine_entry_text(entry) for entries in self.knowledge_base.values() for entry in entries]
                self.tfidf_matrix = self.vectorizer.fit_transform(documents)
                logger.info("✅ TF-IDF vectorizer initialized with existing knowledge base.")
            else:
                self.tfidf_matrix = None
                logger.info("⚠️ Knowledge base is empty. TF-IDF vectorizer not initialized.")
        except Exception as e:
            logger.error(f"❌ Error initializing TF-IDF vectorizer: {e}")

    def _combine_entry_text(self, entry: Dict[str, Any]) -> str:
        """
        📝 Combine relevant text fields of an entry for vectorization.
        """
        return ' '.join([str(value) for key, value in entry.items() if isinstance(value, str)])

    def categorize_data(self, data: List[Dict[str, Any]]) -> None:
        """
        🗂️ Categorize data entries.
        """
        logger.info("📂 Categorizing data")
        for entry in data:
            category = self._determine_category(entry)
            self.knowledge_base[category].append(entry)
            self.categories.add(category)
        logger.info(f"📊 Data categorized into {len(self.categories)} categories")

    def _determine_category(self, entry: Dict[str, Any]) -> str:
        """
        🧩 Determine the category of a data entry.
        """
        if 'type' in entry:
            return entry['type']
        elif 'content' in entry:
            return 'text'
        else:
            return 'uncategorized'

    def summarize_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        📝 Summarize data entries.
        """
        logger.info("✂️ Summarizing data")
        summarized = []
        for entry in data:
            summary = {}
            for key, value in entry.items():
                if isinstance(value, str) and len(value) > 200:
                    summary[key] = (value[:197] + '...') if len(value) > 200 else value
                else:
                    summary[key] = value
            summarized.append(summary)
        logger.info("✅ Data summarization complete")
        return summarized

    def build_knowledge_graph(self) -> None:
        """
        🌐 Build the knowledge graph.
        """
        logger.info("🔗 Building knowledge graph")
        with self.lock:
            self.knowledge_graph.clear()
            for category, entries in self.knowledge_base.items():
                for entry in entries:
                    entry_id = entry.get('id', str(id(entry)))
                    self.knowledge_graph.add_node(entry_id, category=category, name=entry.get('name', ''))
            # Create edges based on shared categories
            for category in self.categories:
                entries = self.knowledge_base[category]
                for i in range(len(entries)):
                    for j in range(i + 1, len(entries)):
                        id1 = entries[i].get('id', str(id(entries[i])))
                        id2 = entries[j].get('id', str(id(entries[j])))
                        self.knowledge_graph.add_edge(id1, id2)
        logger.info("✅ Knowledge graph construction complete")

    def visualize_knowledge_graph(self, output_path: str = "knowledge_graph.png") -> None:
        """
        📈 Visualize the knowledge graph.
        """
        logger.info("🎨 Visualizing knowledge graph")
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.knowledge_graph, k=0.15, iterations=20)
            categories = nx.get_node_attributes(self.knowledge_graph, 'category')
            unique_categories = list(set(categories.values()))
            color_map = {category: plt.cm.tab20(i) for i, category in enumerate(unique_categories)}
            node_colors = [color_map[categories[node]] for node in self.knowledge_graph.nodes()]
            nx.draw(
                self.knowledge_graph,
                pos,
                node_color=node_colors,
                with_labels=True,
                node_size=500,
                font_size=8,
                edge_color='gray',
                alpha=0.7
            )
            plt.title("🌐 Knowledge Graph")
            plt.savefig(output_path)
            plt.close()
            logger.info(f"✅ Knowledge graph visual saved to {output_path}")
        except Exception as e:
            logger.error(f"❌ Error visualizing knowledge graph: {e}")

    def update_tfidf_matrix(self):
        """
        📊 Update the TF-IDF matrix with the current knowledge base.
        """
        logger.info("📊 Updating TF-IDF matrix")
        try:
            documents = [self._combine_entry_text(entry) for entries in self.knowledge_base.values() for entry in entries]
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
            logger.info("✅ TF-IDF matrix updated successfully.")
        except Exception as e:
            logger.error(f"❌ Error updating TF-IDF matrix: {e}")

    def query_knowledge_base(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        🔍 Query the knowledge base using natural language with TF-IDF similarity.
        """
        logger.info(f"🔎 Processing query: {query}")
        if not self.tfidf_matrix:
            logger.warning("⚠️ TF-IDF matrix is not initialized.")
            return []

        try:
            query_vec = self.vectorizer.transform([self._combine_entry_text({'description': query})])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-top_n:][::-1]
            documents = [self._combine_entry_text(entry) for entries in self.knowledge_base.values() for entry in entries]
            top_entries = []
            for idx in top_indices:
                # Find the corresponding entry
                cumulative = 0
                for category, entries in self.knowledge_base.items():
                    if idx < cumulative + len(entries):
                        entry = entries[idx - cumulative]
                        top_entries.append(entry)
                        break
                    cumulative += len(entries)
            logger.info(f"✅ Query returned {len(top_entries)} results")
            return top_entries
        except Exception as e:
            logger.error(f"❌ Error during querying: {e}")
            return []

    def display_summary(self, summarized_data: List[Dict[str, Any]]) -> None:
        """
        📄 Display summarized data in a readable format.
        """
        logger.info("🖨️ Displaying summarized data")
        for entry in summarized_data:
            print(json.dumps(entry, indent=2))

    def save_to_database(self, data: List[Dict[str, Any]]) -> None:
        """
        💾 Save data entries to the database.
        """
        logger.info("💾 Saving data to database")
        try:
            for entry in data:
                entry_id = entry.get('id', str(id(entry)))
                exists_query = self.db_session.query(exists().where(KnowledgeEntry.entry_id == entry_id)).scalar()
                if not exists_query:
                    knowledge_entry = KnowledgeEntry(
                        entry_id=entry_id,
                        name=entry.get('name', ''),
                        category=self._determine_category(entry),
                        description=entry.get('description', '')
                    )
                    self.db_session.add(knowledge_entry)
            self.db_session.commit()
            logger.info("✅ Data saved to database successfully")
            # Update TF-IDF matrix after saving new data
            self.update_tfidf_matrix()
        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"❌ Database error: {e}")
        except Exception as e:
            logger.error(f"❌ Error saving data to database: {e}")

    def load_from_database(self) -> None:
        """
        📥 Load data entries from the database into the knowledge base.
        """
        logger.info("📥 Loading data from database")
        try:
            entries = self.db_session.query(KnowledgeEntry).all()
            data = []
            for entry in entries:
                data.append({
                    'id': entry.entry_id,
                    'name': entry.name,
                    'type': entry.category,
                    'description': entry.description
                })
            self.categorize_data(data)
            self.build_knowledge_graph()
            logger.info("✅ Data loaded from database successfully")
            # Update TF-IDF matrix after loading data
            self.update_tfidf_matrix()
        except SQLAlchemyError as e:
            logger.error(f"❌ Database error: {e}")
        except Exception as e:
            logger.error(f"❌ Error loading data from database: {e}")

    def update_data_source(self, data_sources: List[Dict[str, Any]]) -> None:
        """
        🔄 Re-process data sources to update the knowledge base and database.
        """
        logger.info("🔄 Updating data sources")
        all_data = []
        for ds in data_sources:
            format_type = ds.get('format')
            source = ds.get('source')
            parser = get_parser(format_type)
            if not parser:
                logger.error(f"❌ No parser available for format: {format_type}")
                continue

            if format_type == 'api':
                params = ds.get('params', {})
                data = parser.parse(source, params=params)
            else:
                data = parser.parse(source)

            if data:
                all_data.extend(data)

        if not all_data:
            logger.warning("⚠️ No data parsed from any source during update")
            return

        with self.lock:
            self.knowledge_base.clear()
            self.categories.clear()
            self.knowledge_graph.clear()
            self.categorize_data(all_data)
            self.save_to_database(all_data)
            self.build_knowledge_graph()

    def start_real_time_updates(self, data_sources: List[Dict[str, Any]], watch_paths: List[str]) -> None:
        """
        🕒 Start monitoring specified directories for real-time updates.
        """

        logger.info("🕒 Starting real-time updates monitoring")

        event_handler = DataChangeHandler(self, data_sources)
        observer = Observer()

        for path in watch_paths:
            if os.path.isdir(path):
                observer.schedule(event_handler, path=path, recursive=True)
                logger.info(f"🔍 Watching directory for changes: {path}")
            else:
                logger.warning(f"⚠️ Watch path is not a directory or does not exist: {path}")

        observer_thread = threading.Thread(target=observer.start, daemon=True)
        observer_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            logger.info("🛑 Stopping real-time updates monitoring")

        observer.join()


# =========================
# File System Event Handler for Real-Time Updates
# =========================

class DataChangeHandler(FileSystemEventHandler):
    """
    📂 Handles file system events to trigger data updates.
    """

    def __init__(self, processor: DataProcessor, data_sources: List[Dict[str, Any]]):
        super().__init__()
        self.processor = processor
        self.data_sources = data_sources
        self.debounce_time = 2  # seconds
        self.last_event = time.time()

    def on_modified(self, event):
        current_time = time.time()
        if current_time - self.last_event > self.debounce_time:
            logger.info(f"🔄 Detected modification: {event.src_path}")
            self.processor.update_data_source(self.data_sources)
            self.last_event = current_time

    def on_created(self, event):
        self.on_modified(event)

    def on_deleted(self, event):
        self.on_modified(event)


# =========================
# Flask Web Interface with Authentication
# =========================

app = Flask(__name__)
auth = HTTPBasicAuth()


@auth.verify_password
def verify_password(username, password):
    session = SessionLocal()
    user = session.query(User).filter(User.username == username).first()
    session.close()
    if user and check_password_hash(user.password_hash, password):
        return username
    return None


@app.route('/api/query', methods=['GET'])
@auth.login_required
def api_query():
    query = request.args.get('q', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    processor = app.config['processor']
    results = processor.query_knowledge_base(query)
    return jsonify(results), 200


@app.route('/api/visualize', methods=['GET'])
@auth.login_required
def api_visualize():
    processor = app.config['processor']
    processor.visualize_knowledge_graph()
    return send_file('knowledge_graph.png', mimetype='image/png')


@app.route('/api/entries', methods=['GET'])
@auth.login_required
def get_entries():
    processor = app.config['processor']
    all_entries = []
    with processor.lock:
        for entries in processor.knowledge_base.values():
            all_entries.extend(entries)
    return jsonify(all_entries), 200


@app.route('/api/entries', methods=['POST'])
@auth.login_required
def add_entry():
    processor = app.config['processor']
    entry_data = request.json
    if not entry_data:
        return jsonify({"error": "No data provided"}), 400
    try:
        processor.categorize_data([entry_data])
        processor.save_to_database([entry_data])
        processor.build_knowledge_graph()
        return jsonify({"message": "✅ Entry added successfully"}), 201
    except Exception as e:
        logger.error(f"❌ Error adding entry via API: {e}")
        return jsonify({"error": "Failed to add entry"}), 500


@app.route('/api/entries/<entry_id>', methods=['DELETE'])
@auth.login_required
def delete_entry(entry_id):
    processor = app.config['processor']
    try:
        with processor.lock:
            # Remove from knowledge base
            for category, entries in processor.knowledge_base.items():
                entries[:] = [e for e in entries if e.get('id') != entry_id]
            # Remove from database
            entry = processor.db_session.query(KnowledgeEntry).filter(KnowledgeEntry.entry_id == entry_id).first()
            if entry:
                processor.db_session.delete(entry)
                processor.db_session.commit()
            # Remove from knowledge graph
            if processor.knowledge_graph.has_node(entry_id):
                processor.knowledge_graph.remove_node(entry_id)
        return jsonify({"message": "✅ Entry deleted successfully"}), 200
    except SQLAlchemyError as e:
        processor.db_session.rollback()
        logger.error(f"❌ Database error during deletion: {e}")
        return jsonify({"error": "Database error"}), 500
    except Exception as e:
        logger.error(f"❌ Error deleting entry via API: {e}")
        return jsonify({"error": "Failed to delete entry"}), 500


# =========================
# Command-Line Interface for Queries and Visualizations
# =========================

def cli_interface(processor: DataProcessor):
    """
    🎛️ Interactive command-line interface to interact with the knowledge base.
    """
    logger.info("🚀 Launching CLI Interface. Type 'help' for commands.")
    while True:
        try:
            user_input = input("🔍 >> ").strip()
            if user_input.lower() in ['exit', 'quit']:
                logger.info("🛑 Exiting CLI interface.")
                break
            elif user_input.lower() == 'help':
                print("""
🌟 Available Commands:
  help                Show this help message
  exit, quit          Exit the interface
  query <your query>  Search the knowledge base
  visualize           Generate and view the knowledge graph
  entries             List all knowledge entries
  add                 Add a new knowledge entry
  delete <entry_id>   Delete a knowledge entry by ID
""")
            elif user_input.lower().startswith('query '):
                query = user_input[6:]
                results = processor.query_knowledge_base(query)
                if results:
                    summarized = processor.summarize_data(results)
                    processor.display_summary(summarized)
                else:
                    print("❌ No results found.")
            elif user_input.lower() == 'visualize':
                processor.visualize_knowledge_graph()
                print("📈 Knowledge graph visual saved as 'knowledge_graph.png'.")
            elif user_input.lower() == 'entries':
                all_entries = []
                with processor.lock:
                    for entries in processor.knowledge_base.values():
                        all_entries.extend(entries)
                summarized = processor.summarize_data(all_entries)
                processor.display_summary(summarized)
            elif user_input.lower() == 'add':
                print("📝 Enter new entry details in JSON format:")
                try:
                    entry_input = input(">>> ")
                    entry_data = json.loads(entry_input)
                    processor.categorize_data([entry_data])
                    processor.save_to_database([entry_data])
                    processor.build_knowledge_graph()
                    print("✅ Entry added successfully.")
                except json.JSONDecodeError:
                    print("❌ Invalid JSON format.")
                except Exception as e:
                    logger.error(f"❌ Error adding entry via CLI: {e}")
                    print("❌ Failed to add entry.")
            elif user_input.lower().startswith('delete '):
                entry_id = user_input[7:].strip()
                if not entry_id:
                    print("❌ Please provide an entry ID to delete.")
                    continue
                try:
                    with processor.lock:
                        # Remove from knowledge base
                        for category, entries in processor.knowledge_base.items():
                            entries[:] = [e for e in entries if e.get('id') != entry_id]
                        # Remove from database
                        entry = processor.db_session.query(KnowledgeEntry).filter(KnowledgeEntry.entry_id == entry_id).first()
                        if entry:
                            processor.db_session.delete(entry)
                            processor.db_session.commit()
                        # Remove from knowledge graph
                        if processor.knowledge_graph.has_node(entry_id):
                            processor.knowledge_graph.remove_node(entry_id)
                    print("✅ Entry deleted successfully.")
                except SQLAlchemyError as e:
                    processor.db_session.rollback()
                    logger.error(f"❌ Database error during deletion via CLI: {e}")
                    print("❌ Database error.")
                except Exception as e:
                    logger.error(f"❌ Error deleting entry via CLI: {e}")
                    print("❌ Failed to delete entry.")
            else:
                print("❓ Unknown command. Type 'help' for a list of commands.")
        except Exception as e:
            logger.error(f"❌ Error in CLI interface: {e}")


# =========================
# Main Function
# =========================

def main():
    """
    🏁 Main function to execute the data processing workflow.
    """
    processor = DataProcessor()

    # =========================
    # Define Data Sources
    # =========================
    data_sources = [
        {'format': 'csv', 'source': 'data/sample.csv'},
        {'format': 'json', 'source': 'data/sample.json'},
        {'format': 'text', 'source': 'data/sample.txt'},
        # You can add more data sources here
    ]

    # =========================
    # Create Sample Data if Not Exists
    # =========================
    os.makedirs('data', exist_ok=True)

    sample_csv = 'data/sample.csv'
    if not os.path.isfile(sample_csv):
        with open(sample_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['id', 'name', 'type', 'description'])
            writer.writeheader()
            writer.writerow({
                'id': '1',
                'name': 'Python',
                'type': 'Programming Language',
                'description': 'Python is a versatile high-level programming language known for its readability and broad library support.'
            })
            writer.writerow({
                'id': '2',
                'name': 'JavaScript',
                'type': 'Programming Language',
                'description': 'JavaScript is a dynamic programming language primarily used for enhancing web pages to provide interactive features.'
            })
        logger.info(f"📁 Created sample CSV file: {sample_csv}")

    sample_json = 'data/sample.json'
    if not os.path.isfile(sample_json):
        sample_data = [
            {
                "id": "3",
                "name": "OpenAI",
                "type": "Organization",
                "description": "OpenAI is an AI research and deployment company aiming to ensure that artificial general intelligence benefits all of humanity."
            },
            {
                "id": "4",
                "name": "ChatGPT",
                "type": "AI Model",
                "description": "ChatGPT is a language model developed by OpenAI, capable of understanding and generating human-like text based on the input it receives."
            }
        ]
        with open(sample_json, 'w', encoding='utf-8') as jsonfile:
            json.dump(sample_data, jsonfile, indent=2)
        logger.info(f"📁 Created sample JSON file: {sample_json}")

    sample_text = 'data/sample.txt'
    if not os.path.isfile(sample_text):
        with open(sample_text, 'w', encoding='utf-8') as txtfile:
            txtfile.write("Welcome to CogniSphere, your Intelligent Knowledge Navigator!\nThis text file serves as a sample input for testing the parsing and summarization capabilities of the system.")
        logger.info(f"📁 Created sample text file: {sample_text}")

    # =========================
    # Parse Data Sources
    # =========================
    all_data = []

    for ds in data_sources:
        format_type = ds.get('format')
        source = ds.get('source')
        parser = get_parser(format_type)
        if not parser:
            logger.error(f"❌ No parser available for format: {format_type}")
            continue

        data = parser.parse(source)
        if data:
            all_data.extend(data)

    if not all_data:
        logger.warning("⚠️ No data parsed from any source")
    else:
        processor.categorize_data(all_data)
        processor.save_to_database(all_data)
        processor.build_knowledge_graph()

    # =========================
    # Load Existing Data from Database
    # =========================
    processor.load_from_database()

    # =========================
    # Start Real-Time Updates in a Separate Thread
    # =========================
    watch_paths = ['data']
    update_thread = threading.Thread(target=processor.start_real_time_updates, args=(data_sources, watch_paths), daemon=True)
    update_thread.start()

    # =========================
    # Start Flask Web Server in a Separate Thread
    # =========================
    app.config['processor'] = processor
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True)
    flask_thread.start()
    logger.info("🌐 Flask web server started on http://0.0.0.0:5000")

    # =========================
    # Start CLI Interface
    # =========================
    cli_interface(processor)


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"❌ Unhandled exception in main: {e}", exc_info=True)
        sys.exit(1)


# =========================
# TODOs for Future Enhancements
# =========================
# - 🌟 Implement advanced natural language processing for more accurate and context-aware queries.
# - 🗄️ Integrate with a more robust database system (e.g., PostgreSQL) for enhanced scalability and performance.
# - 📊 Develop a comprehensive web-based dashboard with interactive visualizations for the knowledge base.
# - 🔗 Enhance the knowledge graph with weighted edges, directional relationships, and additional node attributes.
# - 🔒 Add user authentication and role-based access controls for secure and personalized data management.
# - 🤖 Implement machine learning algorithms to automatically categorize, tag, and extract insights from data.
# - ⚡ Optimize performance for handling large datasets, including indexing and caching strategies.
# - 🧪 Create automated unit and integration tests to ensure system reliability and facilitate maintenance.
# - 🌐 Incorporate real-time data streaming from various sources using technologies like Kafka or RabbitMQ.
# - 🧩 Develop plugins or extensions to support additional data formats, sources, and integration with external APIs.
