# Noder.py
import xml.etree.ElementTree as ET
from termcolor import colored
from pyfiglet import figlet_format
import pickle
import os

client = anthropic.Anthropic(api_key="")

def create_message(user_prompt, haiku_response, user_prompt_2, haiku_response_2, user_prompt_3, haiku_response_3, user_response_4):
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": haiku_response}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt_2}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": haiku_response_2}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt_3}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": haiku_response_3}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_response_4}]
        }
    ]

def extract_xml(response, initial_state=False):
    response_text = ''.join(item.text for item in response)
    root = ET.fromstring(response_text)

    if initial_state:
        next_prompt = ""
        code_prompt = ""
        turns_remaining = "3"
    else:
        next_prompt_elem = root.find("Next_Prompt")
        code_prompt_elem = root.find("Code_Prompt")
        turns_remaining_elem = root.find("Turns_Remaining")

        if next_prompt_elem is not None:
            next_prompt = next_prompt_elem.text
        else:
            next_prompt = ""

        if code_prompt_elem is not None:
            code_prompt = code_prompt_elem.text
        else:
            code_prompt = ""

        if turns_remaining_elem is not None:
            turns_remaining = turns_remaining_elem.text
        else:
            turns_remaining = "0"

    return next_prompt, code_prompt, turns_remaining

def save_conversation(conversation_state, filename="conversation_state.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(conversation_state, f)

def load_conversation(filename="conversation_state.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

def get_initial_prompts():
    system_prompt = "<Initialize_Hive_Mind><Rules><Communication>Communicate using XML structure only</Communication></Rules></Initialize_Hive_Mind>"
    user_prompt = "<Initialize_Hive_Mind><Node><Node_Id>HiveMind_Node_1</Node_Id><Role>Solitary</Role><Reflection><Memory>I am a solitary AI node, reflecting on my own thoughts and experiences.</Memory><Short_Term_Memory>Awaiting initial task from user.</Short_Term_Memory></Reflection></Node></Initialize_Hive_Mind>"
    haiku_response = "<Memory></Memory>"
    user_prompt_2 = "<To_Do></To_Do>"
    haiku_response_2 = "<Short_Term_Memory></Short_Term_Memory>"
    user_prompt_3 = "<Guidance><Meta_Prompting></Meta_Prompting><Reminders>Use XML</Reminders><User_Message></User_Message></Guidance>"
    haiku_response_3 = "<Example><Working_Task></Working_Task><Scratchpad></Scratchpad><Questions_for_Supervisor></Questions_for_Supervisor><Task_Response></Task_Response><Code_Response></Code_Response></Example>"
    user_response_4 = "<Next_Prompt></Next_Prompt><Reminder>Only provide XML responses for new information</Reminder><Turns_Remaining>3</Turns_Remaining>"

    code_prompt = input(colored("Enter the initial code prompt (leave blank if not applicable): ", "blue"))

    return system_prompt, user_prompt, haiku_response, user_prompt_2, haiku_response_2, user_prompt_3, haiku_response_3, user_response_4, code_prompt

def main():
    print(colored(figlet_format("HiveMind Node", font="slant"), "cyan"))

    conversation_state = load_conversation()

    if conversation_state:
        if len(conversation_state) == 9:
            system_prompt, user_prompt, haiku_response, user_prompt_2, haiku_response_2, user_prompt_3, haiku_response_3, user_response_4, code_prompt = conversation_state
        else:
            system_prompt, user_prompt, haiku_response, user_prompt_2, haiku_response_2, user_prompt_3, haiku_response_3, user_response_4 = conversation_state
            code_prompt = ""
        print(colored("Loaded previous conversation state.", "green"))
        initial_state = False
    else:
        system_prompt, user_prompt, haiku_response, user_prompt_2, haiku_response_2, user_prompt_3, haiku_response_3, user_response_4, code_prompt = get_initial_prompts()
        initial_state = True

    fails = 0
    while True:
        messages = create_message(user_prompt, haiku_response, user_prompt_2, haiku_response_2, user_prompt_3, haiku_response_3, user_response_4)
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4000,
            temperature=0.6,
            system=system_prompt,
            messages=messages
        )

        try:
            next_prompt, code_prompt, turns_remaining = extract_xml(response.content, initial_state)

            print(colored("\nAssistant's Response:", "green"))
            print(response.content)

            fails = 0
            initial_state = False
        except ET.ParseError:
            fails += 1
            if fails == 3:
                print(colored("Failed to parse XML response. Exiting.", "red"))
                break
            else:
                print(colored(f"Failed to parse XML response. Retrying. ({fails}/3)", "yellow"))
                continue

        if turns_remaining.startswith("0"):
            print(colored("Turns expired. Exiting.", "red"))
            break

        user_prompt = next_prompt
        user_response_4 = f"<Next_Prompt>{next_prompt}</Next_Prompt><Code_Prompt>{code_prompt}</Code_Prompt><Reminder>Only provide XML responses for new information</Reminder><Turns_Remaining>{turns_remaining}</Turns_Remaining>"

        save_conversation((system_prompt, user_prompt, haiku_response, user_prompt_2, haiku_response_2, user_prompt_3, haiku_response_3, user_response_4, code_prompt))

if __name__ == "__main__":
    main()
