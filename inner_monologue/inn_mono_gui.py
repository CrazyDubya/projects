from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QSpinBox, QDoubleSpinBox, QDialog
from inner_monologue import InnerMonologue

class InnerMonologueGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.inner_monologue = InnerMonologue()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Inner Monologue GUI')

        layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        input_label = QLabel('User Input:')
        self.input_edit = QLineEdit()
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_edit)
        layout.addLayout(input_layout)

        iterations_layout = QHBoxLayout()
        iterations_label = QLabel('Number of Iterations:')
        self.iterations_spinbox = QSpinBox()
        self.iterations_spinbox.setMinimum(1)
        self.iterations_spinbox.setMaximum(20)
        iterations_layout.addWidget(iterations_label)
        iterations_layout.addWidget(self.iterations_spinbox)
        layout.addLayout(iterations_layout)

        model_layout = QHBoxLayout()
        model_label = QLabel('Model Type:')
        self.model_edit = QLineEdit('CLAUDE_MODELS')
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_edit)
        layout.addLayout(model_layout)

        model_name_layout = QHBoxLayout()
        model_name_label = QLabel('Model Name:')
        self.model_name_edit = QLineEdit('haiku')
        model_name_layout.addWidget(model_name_label)
        model_name_layout.addWidget(self.model_name_edit)
        layout.addLayout(model_name_layout)

        temperature_layout = QHBoxLayout()
        temperature_label = QLabel('Temperature:')
        self.temperature_spinbox = QDoubleSpinBox()
        self.temperature_spinbox.setMinimum(0.0)
        self.temperature_spinbox.setMaximum(1.0)
        self.temperature_spinbox.setSingleStep(0.1)
        self.temperature_spinbox.setValue(0.7)
        temperature_layout.addWidget(temperature_label)
        temperature_layout.addWidget(self.temperature_spinbox)
        layout.addLayout(temperature_layout)

        max_tokens_layout = QHBoxLayout()
        max_tokens_label = QLabel('Max Tokens:')
        self.max_tokens_spinbox = QSpinBox()
        self.max_tokens_spinbox.setMinimum(1)
        self.max_tokens_spinbox.setMaximum(10000)
        self.max_tokens_spinbox.setValue(4000)
        max_tokens_layout.addWidget(max_tokens_label)
        max_tokens_layout.addWidget(self.max_tokens_spinbox)
        layout.addLayout(max_tokens_layout)

        self.run_button = QPushButton('Run Inner Monologue')
        self.run_button.clicked.connect(self.run_inner_monologue)
        layout.addWidget(self.run_button)

        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        layout.addWidget(self.output_edit)

        self.setLayout(layout)

    class QuestionDialog(QDialog):
        def __init__(self, question):
            super().__init__()
            self.setWindowTitle("Question")
            layout = QVBoxLayout()

            label = QLabel(question)
            layout.addWidget(label)

            self.text_edit = QTextEdit()
            layout.addWidget(self.text_edit)

            button = QPushButton("Submit")
            button.clicked.connect(self.accept)
            layout.addWidget(button)

            self.setLayout(layout)

        def get_response(self):
            return self.text_edit.toPlainText()

    def ask_user_question(self, question):
        dialog = QuestionDialog(question)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.get_response()
        return ""

    def run_inner_monologue(self):
        user_input = self.input_edit.text()
        num_iterations = self.iterations_spinbox.value()
        model_type = self.model_edit.text()
        model_name = self.model_name_edit.text()
        temperature = self.temperature_spinbox.value()
        max_tokens = self.max_tokens_spinbox.value()

        for i in range(num_iterations):
            QApplication.processEvents()

            final_output, total_input_tokens, total_output_tokens, process_output = self.inner_monologue.process_user_input(
                user_input, i + 1, model_type, model_name, temperature, max_tokens
            )
            self.output_edit.setPlainText(process_output)

            questions_from_user = self.inner_monologue.extract_questions_for_user(final_output)
            if questions_from_user:
                user_response = self.ask_user_question("\n".join(questions_from_user))
                self.inner_monologue.pause_for_questions(user_response)

            # Update the GUI with iteration and token counts
            self.output_edit.append(f"\nIteration: {i + 1}/{num_iterations}")
            self.output_edit.append(f"Input Tokens: {total_input_tokens}")
            self.output_edit.append(f"Output Tokens: {total_output_tokens}")

        formatted_final_output = self.inner_monologue.format_output(final_output)
        self.output_edit.setPlainText(process_output + "\n\nFinal Output:\n" + formatted_final_output)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    inner_monologue_gui = InnerMonologueGUI()
    inner_monologue_gui.show()
    sys.exit(app.exec_())
