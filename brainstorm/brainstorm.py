# brainstorm.py
from harmonized_api_wrappers import APIWrapper

def create_new_story():
    story_name = input("Enter a name for your new project: ")
    story_directory = story_name.replace(" ", "_")
    os.makedirs(story_directory, exist_ok=True)
    brainstorm_file = os.path.join(story_directory, "brainstorm.txt")
    sentences_file = os.path.join(story_directory, "sentences.txt")
    return story_directory, brainstorm_file, sentences_file

def gather_initial_info():
    print("Let's start brainstorming your story!")
    title = input("What is the book title? ") or "Aliens visit the White House"
    topic = input("What is the topic of the book? ") or "A humorous sci-fi take on Washington DC dysfunction, Galactic Diplomacy and slice of life of First Daughter"
    setting = input("Where and when does the story take place? ") or "Washington,DC Near Future"
    narrative_pov = input("From whose perspective is the story told? ") or "Third-person omniscient"
    tone_style = input("What tone or style do you want for the book? ") or "Epic and adventurous, Young Adult"
    main_characters = input("Who are the main characters? (Separate names with commas) ").split(',') or ["President", "Zobrix", "FIrst Daughter Alexa", "Alien Kid Zip" ]
    main_themes = input("What are the main themes? ") or "new culures and friends, humerous take on government dysfunction"
    central_conflict = input("What is the central conflict or problem? ") or "Zobrix trying to give USA teleportation technolgy"
    resolution = input("How do you envision the story's resolution or climax? ") or "the kds bring the adults together"
    character_arcs = input("Describe the main characters' arcs or development. ") or "Adults learn they can still learn"
    message = input("Is there a particular message or lesson in the story? ") or "how friendship can change the world"

    sub_plots = input("Any sub-plots? (Briefly describe) ") or "Alien dog chasing squirrel around DC"
    num_chapters = int(input("How many chapters do you want? (Default is 5) ") or "5")

    return title, topic, setting, narrative_pov, tone_style, main_characters, main_themes, central_conflict, resolution, character_arcs, message, sub_plots, num_chapters


def generate_outline(api_wrapper, title, topic, main_characters, main_themes, sub_plots, num_chapters, model_type, model_name, conversation_history):
    system_prompt = "You are an a gifted author and critic. You are helping a user generate a high-level outline for a story."
    user_prompt = f"""Please generate a high-level outline for a story with the following details:
    Title: {title}
    Topic: {topic}
    Main Characters: {', '.join(main_characters)}
    Main Themes: {main_themes}
    Sub-plots: {sub_plots}

    Generate an outline with {num_chapters} chapters, providing a brief title for each chapter.

    Conversation History:
    {conversation_history}"""

    response = api_wrapper.process_model(
        model_type, model_name,
        system_prompt=system_prompt,
        refined_input=user_prompt,
        temperature=0.7,
        max_tokens=3000
    )

    chapters = response.strip().split('\n')
    conversation_history += f"\nUser: {user_prompt}\nAssistant: {response}"
    return chapters, conversation_history

def save_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def append_to_file(file_path, content):
    with open(file_path, 'a') as file:
        file.write(content)

def main():
    api_wrapper = APIWrapper()

    story_directory, brainstorm_file, sentences_file = create_new_story()
    archive_directory = os.path.join(story_directory, "archive")
    os.makedirs(archive_directory, exist_ok=True)
    raw_output_file = os.path.join(archive_directory, "rawoutput.txt")

    (
        title, topic, setting, narrative_pov, tone_style,
        main_characters, main_themes, central_conflict, resolution,
        character_arcs, message, sub_plots, num_chapters
    ) = gather_initial_info()

    model_type = "CLAUDE_MODELS"
    model_name = "haiku"

    conversation_history = ""
    chapters, conversation_history = generate_outline(api_wrapper, title, topic, main_characters, main_themes, sub_plots, num_chapters, model_type, model_name, conversation_history)

    outline_content = "# Outline\n\n"
    for i, chapter in enumerate(chapters, start=1):
        outline_content += f"Chapter {i}: {chapter}\n"

    save_to_file(brainstorm_file, outline_content)
    save_to_file(raw_output_file, conversation_history)

    print("\nInitial Outline:")
    print(outline_content)

    while True:
        refinement = input("Any refinements? (Press Enter to accept the outline): ")
        if not refinement:
            break

        conversation_history += f"\nUser: Please refine the outline based on the following feedback: {refinement}"
        chapters, conversation_history = generate_outline(api_wrapper, title, topic, main_characters, main_themes, sub_plots, num_chapters, model_type, model_name, conversation_history)

        refined_outline_content = "# Refined Outline\n\n"
        for i, chapter in enumerate(chapters, start=1):
            refined_outline_content += f"Chapter {i}: {chapter}\n"

        save_to_file(brainstorm_file, refined_outline_content)
        save_to_file(raw_output_file, conversation_history)

        print("\nRefined Outline:")
        print(refined_outline_content)

    # Update brainstorm.txt with the appropriate structure
    brainstorm_content = "# Brainstorm\n\n"
    for i, chapter in enumerate(chapters, start=1):
        brainstorm_content += f"Chapter {i}: {chapter}\n"
        brainstorm_content += "  Plot Points:\n"
        brainstorm_content += "    1. [Plot Point 1]\n"
        brainstorm_content += "    2. [Plot Point 2]\n"
        brainstorm_content += "    3. [Plot Point 3]\n"
        brainstorm_content += "  Sub-plots:\n"
        brainstorm_content += "    1. [Sub-plot 1]\n"
        brainstorm_content += "    2. [Sub-plot 2]\n"
        brainstorm_content += "    3. [Sub-plot 3]\n"
        brainstorm_content += "  Sentences:\n"
        brainstorm_content += "    - [Sentence 1]\n"
        brainstorm_content += "    - [Sentence 2]\n"
        brainstorm_content += "    - [Sentence 3]\n\n"

    save_to_file(brainstorm_file, brainstorm_content)

    # Generate sentences for each chapter
    sentences_content = ""
    for i, chapter in enumerate(chapters, start=1):
        sentences_content += f"Chapter {i}: {chapter}\n"
        sentences_content += "  Sentences:\n"
        # Generate sentences for the chapter (not implemented in this example)
        sentences_content += "    - [Sentence 1]\n"
        sentences_content += "    - [Sentence 2]\n"
        sentences_content += "    - [Sentence 3]\n\n"

    save_to_file(sentences_file, sentences_content)

    print("Brainstorming session completed. Moving to the next module.")

if __name__ == "__main__":
    main()