import logging
import re
import sys
import xml.etree.ElementTree as ET

import anthropic
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, \
    QTextEdit, QWidget, QTabWidget, QFileDialog


class MetaPrompt:
    def __init__(self, prompt_id, node_type, category, rating, description):
        self.id = prompt_id
        self.node_type = node_type
        self.category = category
        self.rating = rating
        self.description = description

    def to_dict(self):
        return {
            'id': self.id,
            'node_type': self.node_type,
            'category': self.category,
            'rating': self.rating,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            prompt_id=data['id'],
            node_type=data['node_type'],
            category=data['category'],
            rating=data['rating'],
            description=data['description']
        )


class MetaPromptBank:
    def __init__(self):
        self.meta_prompts = []

    def add_meta_prompt(self, meta_prompt):
        self.meta_prompts.append(meta_prompt)

    def get_meta_prompt(self, prompt_id):
        for meta_prompt in self.meta_prompts:
            if meta_prompt.id == prompt_id:
                return meta_prompt
        return None

    def update_meta_prompt(self, prompt_id, updated_meta_prompt):
        for i, meta_prompt in enumerate(self.meta_prompts):
            if meta_prompt.id == prompt_id:
                self.meta_prompts[i] = updated_meta_prompt
                return True
        return False

    def remove_meta_prompt(self, prompt_id):
        for i, meta_prompt in enumerate(self.meta_prompts):
            if meta_prompt.id == prompt_id:
                del self.meta_prompts[i]
                return True
        return False

    def search_meta_prompts(self, node_type=None, category=None, min_rating=None):
        results = []
        for meta_prompt in self.meta_prompts:
            if (node_type is None or meta_prompt.node_type == node_type) and \
                    (category is None or meta_prompt.category == category) and \
                    (min_rating is None or meta_prompt.rating >= min_rating):
                results.append(meta_prompt)
        return results

    def update_ratings(self, prompt_id, new_rating):
        meta_prompt = self.get_meta_prompt(prompt_id)
        if meta_prompt:
            meta_prompt.rating = new_rating
            return True
        return False


class Node(QObject):
    response_received = pyqtSignal(str)

    def __init__(self, node_id, node_type, role, task=None, supervisor=None):
        super().__init__()
        self.node_id = node_id
        self.node_type = node_type
        self.role = role
        self.task = task
        self.supervisor = supervisor
        self.conversation_history = []
        self.final_output = None
        self.meta_prompt_bank = []
        self.related_nodes = []
        # Set up a logger for the Node class instances
        self.logger = logging.getLogger(f"Node_{self.node_id}")
        self.logger.setLevel(logging.DEBUG)  # Set the appropriate logging level
        # Ensure there is at least one handler
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def add_to_conversation_history(self, message):
        self.conversation_history.append(message)

    def set_final_output(self, output):
        self.final_output = output

    def to_xml(self):
        node = ET.Element('node')
        ET.SubElement(node, 'id').text = self.node_id
        ET.SubElement(node, 'role').text = self.role
        ET.SubElement(node, 'supervisor').text = self.supervisor
        ET.SubElement(node, 'task').text = self.task
        if self.related_nodes:
            connections = ET.SubElement(node, 'connections')
            for rel_node in self.related_nodes:
                ET.SubElement(connections, 'node').text = rel_node
        return node

    def initialize_meta_prompt_bank(self, meta_prompts):
        self.meta_prompt_bank = meta_prompts

    def create_meta_prompt(self, prompt_id, category, description):
        meta_prompt = MetaPrompt(prompt_id, self.node_type, category, 0, description)
        self.meta_prompt_bank.append(meta_prompt)
        return meta_prompt

    def modify_meta_prompt(self, prompt_id, category=None, description=None):
        for meta_prompt in self.meta_prompt_bank:
            if meta_prompt.id == prompt_id:
                if category:
                    meta_prompt.category = category
                if description:
                    meta_prompt.description = description
                return True
        return False

    def refine_meta_prompt(self, prompt_id, new_description):
        for meta_prompt in self.meta_prompt_bank:
            if meta_prompt.id == prompt_id:
                meta_prompt.description = new_description
                return True
        return False

    def retrieve_meta_prompts(self, category=None, min_rating=None):
        results = []
        for meta_prompt in self.meta_prompt_bank:
            if (category is None or meta_prompt.category == category) and \
                    (min_rating is None or meta_prompt.rating >= min_rating):
                results.append(meta_prompt)
        return results

    def communicate_with_claude(self, client, leader_task):
        try:
            # Read the node's XML file
            file_path = os.path.join("nodes", f"{self.node_id}.xml")
            with open(file_path, "r") as file:
                xml_content = file.read()

            conversation_history = [
                {
                    "role": "user",
                    "content": f"Node ID: {self.node_id}\nNode Type: {self.node_type}\nRole: {self.role}\nTask: {self.task}\n\nXML Content:\n{xml_content}\n\nLeader Task: {leader_task}"
                }
            ]

            system_message = f"You are a node in a hive mind. Your role is: {self.role}"

            max_iterations = 4
            iteration_count = 0

            while iteration_count < max_iterations:
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=4000,
                    temperature=0.6,
                    system=system_message,
                    messages=conversation_history
                )

                node_response = response.content[0].text
                self.add_to_conversation_history({"role": "assistant", "content": node_response})
                self.logger.info(f"Received response from Claude: {node_response}")

                # Save the model response
                response_file = os.path.join("responses", f"{self.node_id}_{iteration_count}.txt")
                os.makedirs("responses", exist_ok=True)
                with open(response_file, "w") as file:
                    file.write(node_response)

                self.response_received.emit(node_response)

                if "Final Output:" in node_response:
                    final_output = node_response.split("Final Output:")[1].strip()
                    self.set_final_output(final_output)
                    break

                iteration_count += 1

            if iteration_count >= max_iterations:
                self.logger.warning(f"Node {self.node_id} reached the maximum number of iterations.")

            self.logger.info(f"Node {self.node_id} conversation completed.")

        except Exception as e:
            self.logger.error(f"An error occurred during communication with Claude: {e}")
            # Implement fallback mechanisms or error handling here


class HiveMind:
    def __init__(self, goal, nodes_directory):
        self.goal = goal
        self.nodes_directory = nodes_directory
        self.nodes = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.meta_prompt_bank = MetaPromptBank()

    def create_node(self, name, role, supervisor, task, connections=None):
        # Assuming this function creates a Node object and initializes it with provided details
        # This is a simplified example. Implement according to your application's requirements.
        node = Node(name, role, supervisor, task)
        node.connections = connections or []
        return node

    def parse_leader_output(self, leader_response):
        content = leader_response.content[0].text
        try:
            # Attempt to handle and parse XML content
            root = ET.fromstring("<Root>" + content + "</Root>")  # Encapsulate in a root tag for valid XML
            for node in root.findall('.//Node'):
                name = node.find('Name').text
                role = node.find('Role').text
                supervisor = node.find('Supervisor').text
                goal = node.find('Goal').text
                task = node.find('Task').text

                print(f"Node parsed successfully: ID={name}, Role={role}, Supervisor={supervisor}, Task={task}")
                # Additional logic to create or update node objects in your system goes here

        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
        except Exception as e:
            print(f"Error parsing node details: {e}")

    def set_node_attributes(self, node, node_element_text):
        """ Set additional attributes like goals, methods from node XML content """
        hive_goal_match = re.search(r'<hive-goal>([^<]+)</hive-goal>', node_element_text)
        if hive_goal_match:
            node.hive_goal = hive_goal_match.group(1)

        communication_methods_match = re.search(r'<communication-methods>(.*?)</communication-methods>',
                                                node_element_text, re.DOTALL)
        if communication_methods_match:
            methods = re.findall(r'<method>([^<]+)</method>', communication_methods_match.group(1))
            node.communication_methods = methods

        next_prompt_match = re.search(r'<next-prompt>(.*?)</next-prompt>', node_element_text, re.DOTALL)
        if next_prompt_match:
            description_match = re.search(r'<description>([^<]+)</description>', next_prompt_match.group(1))
            trigger_match = re.search(r'<trigger>([^<]+)</trigger>', next_prompt_match.group(1))
            if description_match and trigger_match:
                node.next_prompt_description = description_match.group(1)
                node.next_prompt_trigger = trigger_match.group(1)

        # Example of initializing meta prompts, ensure node_type check is accurate
        if node.node_type == "CharacterDevelopment":
            node.initialize_meta_prompt_bank([
                MetaPrompt("1", "CharacterDevelopment", "Self-Reflection", 0,
                           "Periodically assess the depth, consistency, and relatability of your characters, identifying areas for improvement and growth."),
                MetaPrompt("2", "CharacterDevelopment", "Task-Specific", 0,
                           "Guide the process of developing character backstories, motivations, arcs, and relationships, breaking down the creation process into manageable steps."),
                MetaPrompt("3", "CharacterDevelopment", "Collaboration", 0,
                           "Actively communicate with PlotDevelopment, WorldBuilding, and other relevant nodes to ensure your characters fit seamlessly into the larger narrative and setting."),
                MetaPrompt("4", "CharacterDevelopment", "Inspiration", 0,
                           "Explore diverse character archetypes, draw from real-life experiences and observations, and find innovative ways to make your characters unique and memorable."),
                MetaPrompt("5", "CharacterDevelopment", "Accountability", 0,
                           "Keep on track with character development milestones, ensure you are meeting the needs of the overall project, and maintain a steady pace of progress.")
            ])

    def prepare_claude_communication(self):
        for node in self.nodes:
            # Generate the XML format for the node
            xml_content = ET.tostring(node.to_xml(), encoding='utf-8').decode('utf-8')

            # Save the XML content to a file
            file_path = os.path.join(self.nodes_directory, f"{node.node_id}.xml")
            with open(file_path, "w") as file:
                file.write(xml_content)

    def send_leader_output(self, client):
        system_message = "\n".join([
            "You are the leader node in a hive mind.",
            "",
            "**User Interface**:",
            "- Allows users to input goals for the hive-mind to pursue.",
            "",
            "**Leader Node**:",
            "- An LLM (like an Anthropic Claude model) that interprets user goals, creates and manages nodes, and oversees the hive-mind structure.",
            "",
            "**Subject Nodes**:",
            "- Unique nodes created for each new subject or concept, with an assigned supervisor node.",
            "- Named using the format uniquesequentialnumber.majorsubject.specificsubject.",
            "",
            "**Analysis Nodes**:",
            "- Combine information from multiple subject nodes as needed and report insights to the user or other nodes.",
            "",
            "**Explorer Nodes**:",
            "- Spin off from existing subject nodes to investigate related ideas and create new subject nodes.",
            "",
            "**Index Node**:",
            "- Maintains a registry of all nodes, their relationships, roles, and supervisory structure.",
            "- Updated by the Leader node.",
            "",
            "**API Call Manager**:",
            "- Controls the rate of API calls to ensure no more than 1 call per second to comply with rate limits.",
            "",
            "**Node Communication via XML**:",
            "- Each node has an associated XML file used for inter-node communication and state persistence.",
            "- The XML contains the node's name, role, connections to related nodes, supervisor node, overall hive goal, specific task, and methods for contacting other nodes and reporting to supervisors.",
            "- Nodes are stateless, so the complete relevant history needs to be passed for each inference.",
            "",
            "**Constraints**:",
            "- Maximum output per response is capped at 4,000 tokens.",
            "- Maximum content window for a Claude model context is 200,000 tokens.",
            "- Prioritize overall cohesion of the hive-mind over speed of execution.",
            "",
            "**Implementation Details**:",
            "- Use the provided ApiCallRateLimiter class to manage API call frequency.",
            "- The LeaderNode class interprets goals and manages node creation/delegation.",
            "- WorkerNode and XMLModelInterface classes enable nodes to perform tasks and persist state via XML.",
            "- The leader assigns each node a supervisor upon creation to manage its activities.",
            "- Analysis and Explorer nodes provide ways to synthesize knowledge and expand to new subjects.",
            "",
            "**Execution Flow**:",
            "- User inputs a goal via the interface.",
            "- The Leader Node ingests the goal and determines required subjects/tasks.",
            "- Leader creates Subject Nodes for each key aspect, assigns them supervisors, and logs them in the Index.",
            "- Subject Nodes perform their tasks, engaging Analysis and Explorer Nodes as needed.",
            "- Analysis Nodes aggregate insights and report back up the hierarchy.",
            "- Explorer Nodes create new Subject Nodes as they discover tangential concepts.",
            "- The hive-mind continues to grow and adapt to thoroughly address the goal."
            "- #.word.word is expected to be the naming convention for nodes."
        ])

        messages = [
            {
                "role": "user",
                "content": f"Hive Mind Goal: {self.goal}. Never forget your XML!"
            }
        ]

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0.6,
            system=system_message,
            messages=messages
        )

        print("Received leader response from Claude.")
        print(f"Leader response: {response}")

        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        return response

    def leader_node_communication(self):
        max_iterations = 2
        iteration_count = 0

        while iteration_count < max_iterations:
            for node in self.nodes:
                print(f"Leader communicating with node {node.node_id}")

                # Generate useful next prompts for the node
                if node.role == "Explorer":
                    # Generate prompts for Explorer nodes
                    exploration_prompts = [
                        "Identify related concepts to the current subject and propose new areas for exploration.",
                        "Analyze the potential impact of exploring new subjects on the overall hive mind goal.",
                        "Assess the feasibility and resources required to pursue new exploration paths."
                    ]
                    node.meta_prompt_bank.extend(exploration_prompts)
                elif node.role == "Supervisor":
                    # Generate prompts for Supervisor nodes
                    supervisor_prompts = [
                        "Review the progress and performance of worker nodes under your supervision.",
                        "Identify any bottlenecks or challenges faced by worker nodes and propose solutions.",
                        "Assess the coordination and collaboration among worker nodes and suggest improvements."
                    ]
                    node.meta_prompt_bank.extend(supervisor_prompts)
                else:
                    # Generate prompts for other node types
                    generic_prompts = [
                        "Reflect on your current progress and identify areas for improvement.",
                        "Analyze how your task contributes to the overall hive mind goal.",
                        "Propose ways to optimize your performance and efficiency."
                    ]
                    node.meta_prompt_bank.extend(generic_prompts)

                # Monitor node progress and provide guidance
                progress = node.assess_progress()
                if progress < 0.5:
                    guidance_prompt = "Your progress seems to be slow. Please focus on your primary task and seek assistance from your supervisor if needed."
                    node.meta_prompt_bank.append(guidance_prompt)

                node.communicate_with_claude(self.client, self.goal)

            iteration_count += 1

            # Check if all nodes have completed their tasks
            all_tasks_completed = all(node.final_output is not None for node in self.nodes)
            if all_tasks_completed:
                print("All nodes have completed their tasks. Hive mind goal achieved!")
                break

        if iteration_count >= max_iterations:
            print("Leader Node reached the maximum number of iterations.")

    def save_node_outputs(self):
        for node in self.nodes:
            output_file = os.path.join(self.nodes_directory, f"{node.node_id}_output.txt")
            with open(output_file, "w") as file:
                file.write(f"Node ID: {node.node_id}\n")
                file.write(f"Final Output:\n{node.final_output}\n")


class ProgressTracker:
    def __init__(self):
        self.logs = []

    def log_activity(self, node_id, activity, timestamp):
        log_entry = {
            'node_id': node_id,
            'activity': activity,
            'timestamp': timestamp
        }
        self.logs.append(log_entry)

    def save_logs(self, file_path):
        with open(file_path, 'w') as file:
            for log_entry in self.logs:
                file.write(f"{log_entry['node_id']},{log_entry['activity']},{log_entry['timestamp']}\n")

    def display_progress(self, node_id):
        node_logs = [log for log in self.logs if log['node_id'] == node_id]
        if node_logs:
            latest_activity = node_logs[-1]['activity']
            print(f"Node {node_id} - Latest activity: {latest_activity}")
        else:
            print(f"No progress logs found for Node {node_id}")


class HiveMindGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hive Mind")
        self.setGeometry(100, 100, 1000, 800)
        self.hive_mind_worker = None

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.goal_layout = QHBoxLayout()
        self.goal_label = QLabel("Hive Mind Goal:")
        self.goal_input = QLineEdit()
        self.goal_button = QPushButton("Set Goal")
        self.goal_button.clicked.connect(self.set_goal)
        self.goal_layout.addWidget(self.goal_label)
        self.goal_layout.addWidget(self.goal_input)
        self.goal_layout.addWidget(self.goal_button)
        self.layout.addLayout(self.goal_layout)

        self.load_button = QPushButton("Load from Files")
        self.load_button.clicked.connect(self.load_from_files)
        self.layout.addWidget(self.load_button)

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self.leader_output_tab = QWidget()
        self.leader_output_layout = QVBoxLayout(self.leader_output_tab)
        self.leader_output_label = QLabel("Leader Output:")
        self.leader_output_text = QTextEdit()
        self.leader_output_text.setReadOnly(True)
        self.leader_output_layout.addWidget(self.leader_output_label)
        self.leader_output_layout.addWidget(self.leader_output_text)
        self.tab_widget.addTab(self.leader_output_tab, "Leader Output")

        self.node_list_tab = QWidget()
        self.node_list_layout = QVBoxLayout(self.node_list_tab)
        self.node_list_label = QLabel("Node List:")
        self.node_list_text = QTextEdit()
        self.node_list_text.setReadOnly(True)
        self.node_list_layout.addWidget(self.node_list_label)
        self.node_list_layout.addWidget(self.node_list_text)
        self.tab_widget.addTab(self.node_list_tab, "Node List")

        self.node_communication_tab = QWidget()
        self.node_communication_layout = QVBoxLayout(self.node_communication_tab)
        self.node_communication_label = QLabel("Node Communication:")
        self.node_communication_text = QTextEdit()
        self.node_communication_text.setReadOnly(True)
        self.node_communication_layout.addWidget(self.node_communication_label)
        self.node_communication_layout.addWidget(self.node_communication_text)
        self.tab_widget.addTab(self.node_communication_tab, "Node Communication")

        self.client = anthropic.Anthropic()
        self.hive_mind = None
        self.progress_tracker = ProgressTracker()

        # Add a new tab for progress updates and logs
        self.progress_tab = QWidget()
        self.progress_layout = QVBoxLayout(self.progress_tab)
        self.progress_label = QLabel("Progress Updates:")
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_layout.addWidget(self.progress_label)
        self.progress_layout.addWidget(self.progress_text)
        self.tab_widget.addTab(self.progress_tab, "Progress")

    def set_goal(self):
        self.goal_button.setEnabled(False)
        self.goal_button.setText("Processing...")

        hive_mind_goal = self.goal_input.text()
        nodes_directory = 'nodes'
        os.makedirs(nodes_directory, exist_ok=True)

        hive_mind = HiveMind(hive_mind_goal, nodes_directory)
        leader_response = hive_mind.send_leader_output(self.client)
        self.update_progress(f"Leader response: {leader_response}")

        self.update_progress("Leader output parsing started.")
        hive_mind.parse_leader_output(leader_response)
        self.update_progress("Leader output parsing completed.")

        self.update_progress("Preparing Claude communication for nodes...")
        hive_mind.prepare_claude_communication()
        self.update_progress("Claude communication preparation completed.")

        self.update_progress("Nodes communicating with Claude...")
        for node in hive_mind.nodes:
            node_info = f"Node {node.node_id} - Role: {node.role}, Task: {node.task}"
            self.update_progress(node_info)

            node_response = node.communicate_with_claude(self.client, hive_mind_goal)
            node_response_output = f"Node {node.node_id} received response from Claude:\n{node_response}"
            self.update_progress(node_response_output)

            node_comm_completed_output = f"Node {node.node_id} communication with Claude completed."
            self.update_progress(node_comm_completed_output)

        self.update_progress("Node communication with Claude completed.")

        hive_mind.leader_node_communication()
        hive_mind.save_node_outputs()

        self.goal_button.setEnabled(True)
        self.goal_button.setText("Set Goal")

    def update_progress(self, message):
        self.progress_text.append(message)
        self.save_to_file("progress.txt", message + "\n")

    def load_logs(self, file_path):
        with open(file_path, 'r') as file:
            logs = file.readlines()
        for log in logs:
            node_id, activity, timestamp = log.strip().split(',')
            self.progress_tracker.log_activity(node_id, activity, timestamp)

    def display_logs(self):
        self.progress_text.clear()
        for log_entry in self.progress_tracker.logs:
            self.progress_text.append(
                f"Node {log_entry['node_id']} - {log_entry['activity']} - {log_entry['timestamp']}")

    def save_to_file(self, file_name, content):
        with open(file_name, 'a') as file:
            file.write(content)

    def load_from_files(self):
        leader_output_file, _ = QFileDialog.getOpenFileName(self, "Select Leader Output File", "", "Text Files (*.txt)")
        if leader_output_file:
            with open(leader_output_file, "r") as file:
                leader_output = file.read()
                self.leader_output_text.setText(leader_output)

        node_list_file, _ = QFileDialog.getOpenFileName(self, "Select Node List File", "", "Text Files (*.txt)")
        if node_list_file:
            with open(node_list_file, "r") as file:
                node_list = file.read()
                self.node_list_text.setText(node_list)

        node_communication_file, _ = QFileDialog.getOpenFileName(self, "Select Node Communication File", "",
                                                                 "Text Files (*.txt)")
        if node_communication_file:
            with open(node_communication_file, "r") as file:
                node_communication = file.read()
                self.node_communication_text.setText(node_communication)


def main():
    app = QApplication(sys.argv)
    hive_mind_gui = HiveMindGUI()
    hive_mind_gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
