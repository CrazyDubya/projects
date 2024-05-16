# llmchatroom.py
import json
from datetime import datetime
from queue import Queue
from threading import Thread

# Configuration for LLMs, with API key for the third model.
llm_configs = {
    1: {"name": "Mixtral", "base_url": "http://localhost", "port": 8080},
    2: {"name": "Gwen", "base_url": "http://localhost", "port": 1234},
    3: {"name": "ExternalModel", "base_url": "https://api.perplexity.ai", "port": 82, "api_key": "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"},
    # External model with API key
}

def send_request_to_model(model_id, base_url, port, data, response_queue):
    """Sends a request to the specified model and puts the response in the queue."""
    endpoint = f"{base_url}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # Check if the model is the third one and append the API key if needed
    if model_id == 3:
        api_key = llm_configs[model_id].get("api_key")
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(endpoint, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_queue.put(response.json())
    else:
        print(f"An error occurred for Model {llm_configs[model_id]['name']}: {response.status_code}")

def save_conversation(task, messages):
    """Saves the conversation to a file."""
    filename = f"{task[:15]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, 'w') as file:
        for msg in messages:
            file.write(f"{msg['role']}: {msg['content']}\n")
    print(f"Conversation saved to {filename}")

def main():
    print("Available models: 1. Mixtral, 2. Gwen, 3. ExternalModel")
    selected_models_input = input("Enter model numbers separated by commas (e.g., 1,2,3): ")
    selected_models = [int(x.strip()) for x in selected_models_input.split(',') if x.strip().isdigit()]

    task = input("Enter the initial task/message for the LLMs: ")
    num_turns_str = input("Enter number of turns (2-100, default 10): ")
    num_turns = int(num_turns_str) if num_turns_str.isdigit() and 2 <= int(num_turns_str) <= 100 else 10

    system_prompt_bases = []
    for model_id in selected_models:
        system_prompt_base = input(f"Enter system prompt base for Model {model_id}: ")
        system_prompt_bases.append(system_prompt_base)

    chat_between_llms(task, selected_models, num_turns, system_prompt_bases)

def chat_between_llms(task, selected_models, num_turns, system_prompt_bases):
    """Initiates a conversation between selected LLMs based on a task, with a specified number of turns."""
    messages = []
    token_usage = {"total_tokens": 0}
    response_queues = [Queue() for _ in selected_models]

    for turn in range(num_turns):
        for i, model_id in enumerate(selected_models):
            other_model_ids = [mid for mid in selected_models if mid != model_id]
            other_model_name = llm_configs[other_model_ids[0]]["name"]
            model_name = llm_configs[model_id]["name"]
            base_url = llm_configs[model_id]["base_url"]
            port = llm_configs[model_id]["port"]

            system_prompt = system_prompt_bases[i].format(other_model=other_model_name, task=task,
                                                           num_turns=num_turns - turn)

            if turn == 0:
                user_message = {"role": "user", "content": task}
            else:
                user_message = {"role": "user", "content": messages[-1]["content"]} if messages else {"role": "user", "content": ""}

            data = {
                "messages": [user_message],
                "temperature": 0.7,
                "max_tokens": -1,
                "stream": False,
                "system_prompt": system_prompt
            }

            response_queue = response_queues[i]
            thread = Thread(target=send_request_to_model, args=(model_id, base_url, port, data, response_queue))
            thread.start()
            thread.join(timeout=25)  # Timeout after 10 seconds if the model doesn't respond
            if thread.is_alive():
                print(f"Model {model_name} is not responsive. Skipping.")
                continue

            response = response_queue.get()
            if "choices" in response and response["choices"]:
                message_content = response["choices"][0]["message"]["content"]
                messages.append({"role": model_name, "content": message_content})
                token_usage["total_tokens"] += response["usage"]["total_tokens"]
                print(f"{model_name}: {message_content}")
            else:
                print(f"No response received from Model {model_name}")

    save_conversation(task, messages)
    print("Token usage:", token_usage)

if __name__ == "__main__":
    main()
