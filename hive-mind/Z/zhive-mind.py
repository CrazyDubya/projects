# zhive-mind.py
import os
import sys
import xml.etree.ElementTree as ET
import anthropic
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QWidget, QTabWidget, QFileDialog
from PyQt5.QtCore import QTimer

class Node:
    def __init__(self, node_id, node_type, role, task=None, supervisor=None):
        self.node_id = node_id
        self.node_type = node_type
        self.role = role
        self.task = task
        self.supervisor = supervisor

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

    def save_to_file(self, directory):
        xml_string = ET.tostring(self.to_xml(), encoding='utf-8').decode('utf-8')
        file_path = os.path.join(directory, f"{self.node_id}.xml")
        with open(file_path, 'w') as file:
            file.write(xml_string)

    def communicate_with_claude(self, client):
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4000,
            temperature=0.6,
            system=f"You are a node in a hive mind. Your role is: {self.role}",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Node ID: {self.node_id}\nNode Type: {self.node_type}\nRole: {self.role}\nTask: {self.task}"
                        }
                    ]
                }
            ]
        )
        response = message.content[0].text
        print(f"Node {self.node_id} received response from Claude: {response}")
        return response

class HiveMind:
    def __init__(self, goal, nodes_directory):
        self.goal = goal
        self.nodes_directory = nodes_directory
        self.nodes = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def create_node(self, node_id, node_type, role, task=None):
        node = Node(node_id, node_type, role, task)
        self.nodes.append(node)
        node.save_to_file(self.nodes_directory)
        return node

    def parse_leader_output(self, leader_response):
        if leader_response is None or not hasattr(leader_response, 'content'):
            print("Leader response does not contain 'content'. Skipping parsing.")
            return

        content = leader_response.content[0].text
        lines = content.strip().split('\n')
        for line in lines:
            if line.startswith('- '):
                node_info = line[2:].strip().split(':')
                if len(node_info) > 1:
                    node_id = node_info[0].strip()
                    role = ':'.join(node_info[1:]).strip()
                    task = None
                    if '(' in node_id:
                        node_id, node_type = node_id.split('(')
                        node_id = node_id.strip()
                        node_type = node_type.strip(')')
                    else:
                        node_type = 'Subject'  # Assuming default node type is Subject
                    if 'Supervisor:' in role:
                        role, supervisor = role.split('Supervisor:')
                        role = role.strip()
                        supervisor = supervisor.strip()
                        self.create_node(node_id, node_type, role, task, supervisor)
                    else:
                        self.create_node(node_id, node_type, role, task)
    def prepare_claude_communication(self):
        for node in self.nodes:
            claude_prompt = f"Node ID: {node.node_id}\nNode Type: {node.node_type}\nRole: {node.role}\nTask: {node.task}\n\nHive Mind Goal: {self.goal}"
            file_path = os.path.join(self.nodes_directory, f"{node.node_id}_claude_prompt.txt")
            with open(file_path, 'w') as file:
                file.write(claude_prompt)

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
            "- Named using the format sequentialnumber.majorsubject.specificsubject.",
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
        ])

        messages = [
            {
                "role": "user",
                "content": f"Hive Mind Goal: {self.goal}"
            }
        ]

        response = client.messages.create(
            model="claude-3-opus-20240229",
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

class HiveMindGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hive Mind")
        self.setGeometry(100, 100, 1000, 800)

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

        self.goal_button_timer = QTimer()
        self.goal_button_timer.setInterval(1000)  # 1 second
        self.goal_button_timer.timeout.connect(self.reset_goal_button)

    def set_goal(self):
        self.goal_button.setEnabled(False)
        self.goal_button.setText("Processing...")

        hive_mind_goal = self.goal_input.text()
        nodes_directory = 'nodes'
        os.makedirs(nodes_directory, exist_ok=True)

        self.hive_mind = HiveMind(goal=hive_mind_goal, nodes_directory=nodes_directory)
        leader_response = self.hive_mind.send_leader_output(self.client)
        leader_output = f"Leader response:\n{leader_response}"
        self.leader_output_text.append(leader_output)
        self.save_to_file("leader_output.txt", leader_output)

        leader_parsing_output = "Leader output parsing started."
        self.leader_output_text.append(leader_parsing_output)
        self.save_to_file("leader_output.txt", leader_parsing_output)

        self.hive_mind.parse_leader_output(leader_response)
        leader_parsing_completed_output = "Leader output parsing completed."
        self.leader_output_text.append(leader_parsing_completed_output)
        self.save_to_file("leader_output.txt", leader_parsing_completed_output)

        node_prep_output = "Preparing Claude communication for nodes..."
        self.leader_output_text.append(node_prep_output)
        self.save_to_file("leader_output.txt", node_prep_output)

        self.hive_mind.prepare_claude_communication()
        node_prep_completed_output = "Claude communication preparation completed."
        self.leader_output_text.append(node_prep_completed_output)
        self.save_to_file("leader_output.txt", node_prep_completed_output)

        node_comm_output = "Nodes communicating with Claude..."
        self.leader_output_text.append(node_comm_output)
        self.save_to_file("leader_output.txt", node_comm_output)

        for node in self.hive_mind.nodes:
            node_info = f"Node {node.node_id} - Role: {node.role}, Task: {node.task}"
            self.node_list_text.append(node_info)
            self.save_to_file("node_list.txt", node_info)

            node_response = node.communicate_with_claude(self.client)
            node_response_output = f"Node {node.node_id} received response from Claude:\n{node_response}"
            self.node_communication_text.append(node_response_output)
            self.save_to_file(f"node_{node.node_id}_communication.txt", node_response_output)

            node_comm_completed_output = f"Node {node.node_id} communication with Claude completed."
            self.node_communication_text.append(node_comm_completed_output)
            self.save_to_file("node_communication.txt", node_comm_completed_output)

        node_comm_completed_final_output = "Node communication with Claude completed."
        self.leader_output_text.append(node_comm_completed_final_output)
        self.save_to_file("leader_output.txt", node_comm_completed_final_output)

        self.goal_button_timer.start()

    def reset_goal_button(self):
        self.goal_button.setEnabled(True)
        self.goal_button.setText("Set Goal")
        self.goal_button_timer.stop()

    def save_to_file(self, file_name, content):
        with open(file_name, "a") as file:
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


def main():
    app = QApplication(sys.argv)
    hive_mind_gui = HiveMindGUI()
    hive_mind_gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
