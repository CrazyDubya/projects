import random
import re
import time
import csv
import os
from harmonized_api_wrappers import APIWrapper

class InnerMonologue:
    def __init__(self):
        self.api_wrapper = APIWrapper()
        self.long_memory = []
        self.short_memory = []
        self.request_count = 0
        self.token_count = 0
        self.last_request_time = time.time()
        self.last_token_time = time.time()

        self.max_iterations = 69
        self.system_prompt = "You are an AI assistant that uses XML to organize your thoughts and formulate a plan. Please return your internal thoughts, considerations, and any code you generate within <internal_monologue> tags. Also, include questions for the user within <questions_for_user> tags. The final output should include a complete response with the generated code and explanations. You have 5 iterations."
        self.max_requests_per_minute = 50
        self.max_tokens_per_minute = 100000
        self.delay_after_requests = 9
        self.delay_duration = 61

    def analyze_and_refine_prompt(self, internal_monologue):
        refined_prompt = self.system_prompt + " " + internal_monologue
        if self.long_memory:
            refined_prompt += " " + " ".join(random.sample(self.long_memory, min(1, len(self.long_memory))))
        if self.short_memory:
            refined_prompt += " " + " ".join(random.sample(self.short_memory, min(1, len(self.short_memory))))
        return refined_prompt

    def extract_internal_monologue(self, model_output):
        pattern = "<internal_monologue>(.*?)</internal_monologue>"
        match = re.search(pattern, model_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def extract_questions_for_user(self, model_output):
        pattern = "<questions_for_user>(.*?)</questions_for_user>"
        match = re.search(pattern, model_output, re.DOTALL)
        if match:
            questions = match.group(1).strip().split("\n")
            return [question.strip() for question in questions]
        return []

    def pause_for_questions(self, questions_from_user):
        if questions_from_user:
            return questions_from_user
        return []

    def wait_for_rate_limit(self):
        current_time = time.time()
        if self.request_count >= self.max_requests_per_minute:
            elapsed_time = current_time - self.last_request_time
            if elapsed_time < 60:
                wait_time = 60 - elapsed_time
                print(f"Request rate limit exceeded. Waiting for {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            self.last_request_time = time.time()
            self.request_count = 0

        if self.token_count >= self.max_tokens_per_minute:
            elapsed_time = current_time - self.last_token_time
            if elapsed_time < 60:
                wait_time = 60 - elapsed_time
                print(f"Token rate limit exceeded. Waiting for {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            self.last_token_time = time.time()
            self.token_count = 0

    def process_user_input(self, user_input, num_iterations, model_type="CLAUDE_MODELS",
                           model_name="haiku", temperature=0.7, max_tokens=4000):
        if num_iterations > self.max_iterations:
            print(f"Error: The number of iterations exceeds the maximum limit of {self.max_iterations}.")
            return None, 0, 0, ""

        refined_input = user_input
        total_input_tokens = 0
        total_output_tokens = 0

        output_lines = []

        for i in range(num_iterations):
            output_lines.append(f"Iteration {i + 1} - Sending request to the model...")
            self.wait_for_rate_limit()

            response = self.api_wrapper.process_model(
                model_type, model_name,
                system_prompt=self.system_prompt,
                refined_input=refined_input,
                temperature=temperature,
                max_tokens=max_tokens
            )

            model_output = response
            input_tokens = 0  # Placeholder value, replace with actual input token count
            output_tokens = 0  # Placeholder value, replace with actual output token count

            self.request_count += 1
            self.token_count += input_tokens + output_tokens
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            output_lines.append(f"Iteration {i + 1} - Model output received:")
            output_lines.append(model_output)
            output_lines.append("")

            questions_from_user = self.extract_questions_for_user(model_output)
            if questions_from_user and (i == num_iterations // 2 - 1 or i == num_iterations - 1):
                self.pause_for_questions(questions_from_user)

            internal_monologue = self.extract_internal_monologue(model_output)
            self.long_memory.append(internal_monologue)
            self.short_memory.append(internal_monologue)
            self.short_memory = self.short_memory[-5:]

            self.system_prompt = self.analyze_and_refine_prompt(internal_monologue)
            refined_input = model_output

            if (i + 1) % self.delay_after_requests == 0:
                output_lines.append(
                    f"Pausing for {self.delay_duration} seconds after {self.delay_after_requests} requests...")
                time.sleep(self.delay_duration)

        output_lines.append("Generating final output...")
        final_system_prompt = f"{self.system_prompt}\nBased on the refined input and internal monologue, provide a complete and detailed response, including any generated code and explanations. Make sure a completed code block is included"
        final_messages = [{"role": "user", "content": refined_input}]
        self.wait_for_rate_limit()

        final_response = self.api_wrapper.process_model(
            "CLAUDE_MODELS", "sonnet",
            system_prompt=final_system_prompt,
            refined_input=refined_input,
            temperature=0,
            max_tokens=4000
        )

        final_output = final_response
        input_tokens = 0  # Placeholder value, replace with actual input token count
        output_tokens = 0  # Placeholder value, replace with actual output token count

        self.request_count += 1
        self.token_count += input_tokens + output_tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        output_lines.append("Final output:")
        output_lines.append(final_output)
        output_lines.append("")

        process_output = "\n".join(output_lines)

        return final_output, total_input_tokens, total_output_tokens, process_output

    def format_output(self, output):
        if not output:
            return ""

        list_pattern = re.compile(r"- (.*)")
        output = list_pattern.sub(r"• \1", output)

        code_blocks = []

        def replace_code_block(match):
            code_blocks.append(match.group(0))
            return "\0"

        output, _ = re.subn(r"```python.*?```", replace_code_block, output, flags=re.DOTALL)
        output = re.sub(r"`(.*?)`", r"<code>\1</code>", output)
        output = output.replace("\n", "<br>")
        output = output.replace("```python", "<pre><code class='language-python'>").replace("```", "</code></pre>")

        for code_block in code_blocks:
            output = output.replace("\0", code_block, 1)

        output = output.replace("- ", "• ")
        return output
    def get_next_seq(self, file_path):
        try:
            with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                last_line = list(csv.reader(file))[-1]
            return int(last_line[0]) + 1
        except (IOError, IndexError):
            return 1

    def append_to_history(self, seq, max_iterations, system_prompt, user_input, user_iterations, final_output, total_input_tokens, total_output_tokens, model_initial, model_final, temperature_initial, temperature_final):
        file_path = 'history_llm.csv'
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['ID', 'Max Iterations', 'Initial System Prompt', 'User Input', 'User Iterations', 'Final Output', 'Total Input Tokens', 'Total Output Tokens', 'Model Initial', 'Model Final', 'Temperature Initial', 'Temperature Final'])
            writer.writerow([seq, max_iterations, system_prompt, user_input, user_iterations, final_output, total_input_tokens, total_output_tokens, model_initial, model_final, temperature_initial, temperature_final])

    def run(self, user_input, num_iterations, model_type="CLAUDE_MODELS", model_name="haiku",
            temperature=0.7, max_tokens=4000):
        final_output, total_input_tokens, total_output_tokens, process_output = self.process_user_input(
            user_input, num_iterations, model_type, model_name, temperature, max_tokens
        )

        formatted_final_output = self.format_output(final_output)

        file_path = 'history_llm.csv'
        seq = self.get_next_seq(file_path)
        self.append_to_history(
            seq, self.max_iterations, self.system_prompt, user_input, num_iterations, formatted_final_output,
            total_input_tokens, total_output_tokens, f"{model_type}-{model_name}", "CLAUDE_MODELS-sonnet",
            temperature, 0
        )

        return formatted_final_output, total_input_tokens, total_output_tokens, process_output

if __name__ == "__main__":
    inner_monologue = InnerMonologue()
    user_input = input("Enter your input: ")
    num_iterations = int(input("Enter the number of feedback loop iterations: "))
    final_output, total_input_tokens, total_output_tokens = inner_monologue.run(
        user_input, num_iterations, model_type="CLAUDE_MODELS", model_name="haiku", temperature=0.7, max_tokens=4000
    )
    print("Final Output:", final_output)
    print(f"Total input tokens: {total_input_tokens}, Total output tokens: {total_output_tokens}")
