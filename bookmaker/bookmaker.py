# bookmaker.py
import time
from harmonized_api_wrappers import APIWrapper


def gather_initial_info():
    print("Let's start creating your book!")
    title = input("What is the book title? ")
    topic = input("What is the topic of the book? ")
    main_characters = input("Who are the main characters? (Separate names with commas) ").split(',')
    main_themes = input("What are the main themes? ")
    sub_plots = input("Any sub-plots? (Briefly describe) ")
    num_chapters = int(input("How many chapters do you want? (Default is 25) ") or "25")
    model_type = input("Select the model type for generating the book (default: CLAUDE_MODELS): ") or "CLAUDE_MODELS"
    model_name = input("Select the model name for generating the book (default: haiku): ") or "haiku"
    return title, topic, main_characters, main_themes, sub_plots, num_chapters, model_type, model_name


def generate_outline(api_wrapper, topic, main_characters, main_themes, sub_plots, num_chapters, model_type, model_name,
                     conversation_history):
    system_prompt = "You are an AI assistant named Claude. You are helping a user generate a high-level outline for a book."
    user_prompt = f"""Please generate a high-level outline for a book with the following details:
    Topic: {topic}
    Main Characters: {', '.join(main_characters)}
    Main Themes: {main_themes}
    Sub-plots: {sub_plots}

    Generate an outline with {num_chapters} chapters, providing a brief title for each chapter.

    Conversation History:
    {conversation_history}"""

    time.sleep(2)  # Pause for 2 seconds before making the API call
    print("Generating book outline...")
    response = api_wrapper.process_model(model_type, model_name, system_prompt=system_prompt, refined_input=user_prompt,
                                         temperature=0.7, max_tokens=1000)
    chapters = response.strip().split('\n')
    conversation_history += f"\nUser: {user_prompt}\nAssistant: {response}"
    return chapters, conversation_history


def develop_chapter(api_wrapper, chapter_num, chapter_title, model_type, model_name, conversation_history):
    print(f"\nDeveloping Chapter {chapter_num}: {chapter_title}")
    plot_points = int(input(f"How many plot points for Chapter {chapter_num}? "))

    chapter_content = f"Chapter {chapter_num}: {chapter_title}\n\n"

    for i in range(plot_points):
        system_prompt = "You are an AI assistant named Claude. You are helping a user develop a chapter in their book."
        user_prompt = f"""Develop Plot Point {i + 1} for Chapter {chapter_num}: {chapter_title}.
        Provide a detailed description of the plot point.

        Conversation History:
        {conversation_history}"""

        time.sleep(2)  # Pause for 2 seconds before making the API call
        print(f"Developing plot point {i + 1} for chapter {chapter_num}...")
        response = api_wrapper.process_model(model_type, model_name, system_prompt=system_prompt,
                                             refined_input=user_prompt, temperature=0.7, max_tokens=200)
        plot_point = response.strip()
        chapter_content += f"Plot Point {i + 1}: {plot_point}\n"
        conversation_history += f"\nUser: {user_prompt}\nAssistant: {plot_point}"
        chapter_content += develop_plot_point(api_wrapper, chapter_num, chapter_title, i + 1, plot_point, model_type,
                                              model_name, conversation_history)

    return chapter_content, conversation_history


def develop_plot_point(api_wrapper, chapter_num, chapter_title, plot_point_num, plot_point, model_type, model_name,
                       conversation_history, depth=3):
    if depth == 0:
        return ""

    plot_point_content = ""

    for i in range(3):
        system_prompt = "You are an AI assistant named Claude. You are helping a user develop sub-points for a plot point in their book."
        user_prompt = f"""Develop Sub-point {i + 1} for Plot Point {plot_point_num} in Chapter {chapter_num}: {chapter_title}.
        Based on the plot point: {plot_point}
        Provide a detailed description of the sub-point.

        Conversation History:
        {conversation_history}"""

        time.sleep(2)  # Pause for 2 seconds before making the API call
        print(f"Developing sub-point {i + 1} for plot point {plot_point_num} in chapter {chapter_num}...")
        response = api_wrapper.process_model(model_type, model_name, system_prompt=system_prompt,
                                             refined_input=user_prompt, temperature=0.7, max_tokens=100)
        sub_point = response.strip()
        plot_point_content += f"  Sub-point {i + 1}: {sub_point}\n"
        conversation_history += f"\nUser: {user_prompt}\nAssistant: {sub_point}"

        system_prompt = "You are an AI assistant named Claude. You are helping a user elaborate on a sub-point in their book."
        user_prompt = f"""Generate 2-3 sentences elaborating on Sub-point {i + 1} for Plot Point {plot_point_num} in Chapter {chapter_num}: {chapter_title}.
        Sub-point: {sub_point}

        Conversation History:
        {conversation_history}"""

        time.sleep(2)  # Pause for 2 seconds before making the API call
        print(f"Elaborating on sub-point {i + 1} for plot point {plot_point_num} in chapter {chapter_num}...")
        response = api_wrapper.process_model(model_type, model_name, system_prompt=system_prompt,
                                             refined_input=user_prompt, temperature=0.7, max_tokens=100)
        sentences = response.strip()
        plot_point_content += f"    {sentences}\n"
        conversation_history += f"\nUser: {user_prompt}\nAssistant: {sentences}"

    depth -= 1
    plot_point_content += develop_plot_point(api_wrapper, chapter_num, chapter_title, plot_point_num, plot_point,
                                             model_type, model_name, conversation_history, depth)

    return plot_point_content


def save_book_to_file(title, content):
    filename = f"{title}.txt"
    counter = 1
    while os.path.exists(filename):
        filename = f"{title}_{counter}.txt"
        counter += 1

    with open(filename, 'w') as file:
        file.write(content)

    print(f"Book saved to {filename}")


def main():
    api_wrapper = APIWrapper()

    title, topic, main_characters, main_themes, sub_plots, num_chapters, model_type, model_name = gather_initial_info()

    conversation_history = ""
    chapters, conversation_history = generate_outline(api_wrapper, topic, main_characters, main_themes, sub_plots,
                                                      num_chapters, model_type, model_name, conversation_history)

    book_content = f"Title: {title}\n\n"

    for i, chapter in enumerate(chapters, start=1):
        print(f"Developing chapter {i}: {chapter}...")
        chapter_content, conversation_history = develop_chapter(api_wrapper, i, chapter, model_type, model_name,
                                                                conversation_history)
        book_content += f"{chapter_content}\n"

    print("Saving book to file...")
    save_book_to_file(title, book_content)
    print("Your book is complete!")


if __name__ == "__main__":
    main()