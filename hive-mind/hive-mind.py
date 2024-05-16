# hive-mind.py
import sys
import logging
import xml.etree.ElementTree as ET
import anthropic
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QWidget, QTabWidget, QFileDialog
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QThread, Qt
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

        # Configure logging
        self.logger = logging.getLogger(f"Node_{self.node_id}")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(f"node_{self.node_id}.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.nodes = []  # Assuming the nodes are managed within each node instance
        self.nodes_directory = "nodes"  # Default or parameterize this

    def add_to_conversation_history(self, message):
        self.conversation_history.append(message)

    def set_final_output(self, output):
        self.final_output = output
    def to_xml(self):
        node = ET.Element('node', {'id': self.node_id, 'type': self.node_type})
        ET.SubElement(node, 'role').text = self.role
        if self.task:
            ET.SubElement(node, 'task').text = self.task
        if self.supervisor:
            ET.SubElement(node, 'supervisor').text = self.supervisor
        return node

    def create_node(self, node_id, node_type, role, task=None, supervisor=None):
        node = Node(node_id, node_type, role, task, supervisor)
        self.nodes.append(node)
        node.save_to_file(self.nodes_directory)
        return node

    def save_to_file(self, file_name, content, mode='w'):
        file_path = os.path.join(self.hive_mind_directory, file_name)
        try:
            with open(file_path, mode) as file:
                file.write(content + "\n")
        except IOError as e:
            print(f"Error writing to file '{file_path}': {e}")
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

    def assess_progress(self):
        # Example implementation assuming 'progress' attribute exists which could be
        # defined based on specific logic applicable to the node's task and current state
        # For now, we return a simple placeholder value.

        # Placeholder: Let's assume progress is calculated and updated regularly elsewhere in the code
        progress = getattr(self, 'progress', 0)  # Default to 0 if not set
        return progress

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

            max_iterations = 10
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
                os.makedirs(os.path.dirname(response_file), exist_ok=True)
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
            return node_response

        except Exception as e:
            self.logger.error(f"An error occurred during communication with Claude: {e}")
            return None
class HiveMind:

    def __init__(self, goal, nodes_directory):
        self.goal = goal
        self.nodes_directory = nodes_directory
        self.nodes = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.meta_prompt_bank = MetaPromptBank()

    def create_node(self, node_id, node_type, role, task=None, supervisor=None):

        # Read the sample.xml file
        with open("sample.xml", "r") as file:
            xml_template = file.read()

        # Create a new node using the XML template
        node = Node(node_id, node_type, role, task, supervisor)
        node.xml = xml_template.format(node_id=node_id, node_type=node_type, role=role, task=task)
        self.nodes.append(node)
        # Save the node's XML representation to a file
        node.save_to_file(self.nodes_directory)

        return node

    def parse_leader_output(leader_response):
        content = leader_response.content[0].text
        node_pattern = re.compile(r'<node>(.*?)</node>', re.DOTALL)
        node_elements = node_pattern.findall(content)

        for node_element_text in node_elements:
            try:
                name_match = re.search(r'<name>([\d\.]+[\w\.]+)</name>', node_element_text)
                role_match = re.search(r'<role>(.*?)</role>', node_element_text)
                supervisor_match = re.search(r'<supervisor>([\d\.]*[\w\.]*)</supervisor>', node_element_text)
                goal_match = re.search(r'<goal>(.*?)</goal>', node_element_text)
                task_match = re.search(r'<task>(.*?)</task>', node_element_text, re.DOTALL)
                connections_matches = re.findall(r'<connection>([\d\.]+[\w\.]+)</connection>', node_element_text)

                if name_match and role_match and supervisor_match and goal_match and task_match:
                    node_id = name_match.group(1)
                    role = role_match.group(1)
                    supervisor = supervisor_match.group(1)
                    goal = goal_match.group(1).strip()
                    task = task_match.group(1).strip()
                    connections = connections_matches

                    # Successfully parsed node information
                    print(
                        f"Node ID: {node_id}, Role: {role}, Supervisor: {supervisor}, Goal: {goal}, Task: {task}, Connections: {connections}")
                    # Here, the node would be created or updated in your system.
                else:
                    print(f"Failed to parse some node details in: {node_element_text}")

            except Exception as e:
                print(f"Exception during node parsing: {e}\nNode details: {node_element_text}")

    def prepare_claude_communication(self):
        for node in self.nodes:
            # Generate the XML format for the node
            xml_content = node.xml.format(node_id=node.node_id, node_type=node.node_type, role=node.role,
                                          task=node.task)

            # Save the XML content to a file
            file_path = os.path.join(self.nodes_directory, f"{node.node_id}.xml")
            with open(file_path, "w") as file:
                file.write(xml_content)

    def send_leader_output(self, client):
        try:
            with open('leader_output.txt', 'r') as file:
                system_message = file.read()

            messages = [{"role": "user", "content": f"Hive Mind Goal: {self.goal}. Never forget your XML!"}]

            print("Sending request to Claude for leader output...")
            logging.info("Sending request to Claude for leader output")
            print("Messages:", messages)
            print("System Message:", system_message)
            print("Sending leader output API request...")
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.6,
                system=system_message,
                messages=messages
            )
            print("Leader output API response received.")
            print("Received response from Claude for leader output.")
            logging.info("Received response from Claude for leader output")

            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens

            print("Leader response content:", response.content)
            logging.info(f"Leader response content: {response.content}")
            return response.content
        except Exception as e:
            print(f"Error sending leader output: {e}")
            logging.error(f"Error sending leader output: {e}")
            return None
    def leader_node_communication(self):
        max_iterations = 5
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
        os.makedirs(self.nodes_directory, exist_ok=True)  # Ensure the directory exists
        for node in self.nodes:
            output_file = os.path.join(self.nodes_directory, f"{node.node_id}_output.txt")
            try:
                with open(output_file, "w") as file:
                    file.write(f"Node ID: {node.node_id}\n")
                    if node.final_output is not None:
                        file.write(f"Final Output:\n{node.final_output}\n")
                    else:
                        file.write("Final Output: No output recorded.\n")
            except IOError as e:
                # Logging the error can be more sophisticated depending on the setup
                print(f"Failed to write output for node {node.node_id}: {e}")


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

class HiveMindWorker(QThread):
    progress_update = pyqtSignal(str)

    def __init__(self, hive_mind_goal, nodes_directory, client):
        super().__init__()
        self.hive_mind_goal = hive_mind_goal
        self.nodes_directory = nodes_directory
        self.client = client

    def run(self):
        try:
            hive_mind = HiveMind(self.hive_mind_goal, self.nodes_directory)
            self.progress_update.emit("Sending leader output API request...")
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.6,
                system=self.system_message(),
                messages=self.messages()
            )
            if not response or 'content' not in response:
                self.progress_update.emit("Failed to get valid response from API.")
                return

            self.progress_update.emit("API response received.")

            leader_response_content = response['content']
            self.progress_update.emit(f"Leader response received: {leader_response_content}")

            hive_mind.parse_leader_output(leader_response_content)
            self.progress_update.emit("Leader output parsing completed.")

            hive_mind.prepare_claude_communication()
            self.progress_update.emit("Claude communication preparation completed.")

            for node in hive_mind.nodes:
                node_info = f"Node {node.node_id} - Role: {node.role}, Task: {node.task}"
                self.progress_update.emit(node_info)

                node_response = node.communicate_with_claude(self.client, self.hive_mind_goal)
                node_response_output = f"Node {node.node_id} received response from Claude:\n{node_response}"
                self.progress_update.emit(node_response_output)

                node_comm_completed_output = f"Node {node.node_id} communication with Claude completed."
                self.progress_update.emit(node_comm_completed_output)

            hive_mind.leader_node_communication()
            hive_mind.save_node_outputs()
        except Exception as e:
            self.progress_update.emit(f"Error during processing: {str(e)}")


class HiveMindGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hive Mind")
        self.setGeometry(100, 100, 500, 400)
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
        self.hive_mind_directory = "hive_mind_data"
        os.makedirs(self.hive_mind_directory, exist_ok=True)

        self.goal_button_timer = QTimer()
        self.goal_button_timer.setInterval(1000)  # 1 second
        self.goal_button_timer.timeout.connect(self.reset_goal_button)
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
        nodes_directory = os.path.join(self.hive_mind_directory, 'nodes')
        os.makedirs(nodes_directory, exist_ok=True)

        self.hive_mind_worker = HiveMindWorker(hive_mind_goal, nodes_directory, self.client)
        self.hive_mind_worker.progress_update.connect(self.update_progress, Qt.QueuedConnection)
        self.hive_mind_worker.finished.connect(self.on_worker_finished)  # Add this line
        self.hive_mind_worker.start()

        self.goal_button_timer.start()

    def on_worker_finished(self):
        print("HiveMindWorker finished.")

    def update_progress(self, message):
        print(f"Received progress update: {message}")
        self.progress_text.append(message)
        self.progress_text.repaint()
        self.save_to_file(os.path.join(self.hive_mind_directory, "progress_updates.txt"), message, mode='a')

        if "Leader response:" in message:
            if "None" not in message:
                self.leader_output_text.append(message)
                self.save_to_file(os.path.join(self.hive_mind_directory, "leader_output.txt"), message, mode='a')
        elif "Node" in message:
            self.node_communication_text.append(message)
            self.save_to_file(os.path.join(self.hive_mind_directory, "node_communication.txt"), message, mode='a')
            if "Role:" in message:
                self.node_list_text.append(message)
                self.save_to_file(os.path.join(self.hive_mind_directory, "node_list.txt"), message, mode='a')
    def reset_goal_button(self):
        self.goal_button.setEnabled(True)
        self.goal_button.setText("Set Goal")
        self.goal_button_timer.stop()

        if self.hive_mind_worker and self.hive_mind_worker.isRunning():
            self.hive_mind_worker.terminate()
            self.hive_mind_worker.wait()

    def load_logs(self, file_path):
        with open(file_path, 'r') as file:
            logs = file.readlines()
        for log in logs:
            node_id, activity, timestamp = log.strip().split(',')
            self.progress_tracker.log_activity(node_id, activity, timestamp)

    def display_logs(self):
        self.progress_text.clear()
        for log_entry in self.progress_tracker.logs:
            self.progress_text.append(f"Node {log_entry['node_id']} - {log_entry['activity']} - {log_entry['timestamp']}")

    def save_to_file(self, file_name, content, mode='w'):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, mode) as file:
            file.write(content + "\n")

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

    def start_hive_mind(self, hive_mind_goal, nodes_directory, client):
        self.hive_mind_worker = HiveMindWorker(hive_mind_goal, nodes_directory, client)
        self.hive_mind_worker.progress_update.connect(self.update_progress_text)
        self.hive_mind_worker.start()

    def update_progress_text(self, message):
        self.progress_text.append(message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    hive_mind_gui = HiveMindGUI()
    hive_mind_gui.show()
    sys.exit(app.exec_())
