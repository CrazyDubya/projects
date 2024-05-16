# nomic.py
import time
import json
import random
from transformers import BertTokenizer, BertModel
import torch

client = anthropic.Anthropic(api_key="key")

def summarize_history(history, min_reduction=0.5):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize the history
    tokens = tokenizer.tokenize(history)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Create token type IDs and attention mask
    token_type_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)

    # Convert to PyTorch tensors
    input_ids = torch.tensor([input_ids])
    token_type_ids = torch.tensor([token_type_ids])
    attention_mask = torch.tensor([attention_mask])

    # Run the Bert model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs[0]

    # Extract sentence embeddings
    sentence_embeddings = hidden_states[:, 0, :].squeeze().detach().numpy()

    # Calculate sentence scores based on embeddings
    sentence_scores = sentence_embeddings.mean(axis=1)

    # Sort sentences by scores in descending order
    sorted_indices = sentence_scores.argsort()[::-1]

    # Determine the target number of sentences to keep
    num_sentences = len(tokenizer.sent_tokenize(history))
    target_num_sentences = max(1, int(num_sentences * (1 - min_reduction)))

    # Select top sentences based on scores
    selected_indices = sorted(sorted_indices[:target_num_sentences])
    selected_sentences = [tokenizer.sent_tokenize(history)[i] for i in selected_indices]

    # Join selected sentences to form the summary
    summary = ' '.join(selected_sentences)

    return summary


def send_intro_message(llm_id):
    characters = [
        {"name": "Alice", "trait": "Analytical strategist"},
        {"name": "Bob", "trait": "Bold risk-taker"},
        {"name": "Charlie", "trait": "Charismatic negotiator"},
        {"name": "Diana", "trait": "Deceptive tactician"},
        {"name": "Ella", "trait": "Empathetic mediator"},
        {"name": "Frank", "trait": "Fast-paced decision-maker"},
        {"name": "Grace", "trait": "Graceful under pressure"},
        {"name": "Henry", "trait": "Honest and straightforward"},
        {"name": "Iris", "trait": "Innovative thinker"},
        {"name": "Jack", "trait": "Jovial collaborator"},
        {"name": "Kate", "trait": "Keen observer"},
        {"name": "Liam", "trait": "Logical and methodical"},
        {"name": "Mia", "trait": "Master of persuasion"},
        {"name": "Nate", "trait": "Nimble adapter"},
        {"name": "Olivia", "trait": "Optimistic visionary"},
        {"name": "Peter", "trait": "Persistent and determined"},
        {"name": "Quinn", "trait": "Quick-witted improviser"},
        {"name": "Rachel", "trait": "Resourceful problem-solver"},
        {"name": "Sam", "trait": "Silent but impactful"},
        {"name": "Tina", "trait": "Thorough planner"},
        {"name": "Uma", "trait": "Unorthodox strategist"},
        {"name": "Victor", "trait": "Versatile team player"},
        {"name": "Wendy", "trait": "Wise decision-maker"},
        {"name": "Xavier", "trait": "eXceptional under pressure"},
        {"name": "Yvonne", "trait": "Yielding when necessary"},
        {"name": "Zack", "trait": "Zealous competitor"}
    ]

    selected_character = random.choice(characters)
    characters.remove(selected_character)

    intro_message = f"<meta>In this game of Nomic, you embody a unique character with specific strengths and strategic approaches. As players, you are tasked with navigating an evolving landscape of rules, leveraging your character's strengths to propose and vote on rule changes to advance your position in the game. The ultimate goal is to be the first to reach 150 points, achieved through inventive strategies, rule amendments, or by exploiting loopholes in the current rule set.</meta>"
    intro_message += f"You are player {llm_id}, playing as {selected_character['name']}, the {selected_character['trait']}. Keep track of game state, rules, points, and history.</meta>"
    intro_message += f"""
    <syntheticdata>
      <gamestate>
        <turn>5</turn>
        <players>
          <player name="Alice" points="30"/>
          <player name="Bob" points="25"/>
          <player name="Carol" points="35"/>
        </players>
        <rules>
          <rule id="101" mutable="false">Players take turns proposing and voting on rule changes.</rule>
          <rule id="102" mutable="true">Gain 10 points for each proposal that is adopted.</rule>
          <rule id="103" mutable="true">Lose 5 points for each proposal that is rejected.</rule>
        </rules>
        <prevturn>
          <proposed_rules>
            <rule player="Alice">104: Each player starts their turn by drawing a card from a standard deck. Hearts = 5 points, Spades = 10 points.</rule>
            <rule player="Bob">105: Players can trade points with each other during their turn, up to a maximum of 15 points per trade.</rule>
          </proposed_rules>
          <votes>
            <vote player="Alice">Aye, Nay</vote>
            <vote player="Bob">Nay, Aye</vote>
            <vote player="Carol">Aye, Nay</vote>
          </votes>
        </prevturn>
      </gamestate>

      <gamestate>
        <turn>12</turn>
        <players>
          <player name="David" points="80"/>
          <player name="Eve" points="75"/>
          <player name="Frank" points="85"/>
          <player name="Grace" points="90"/>
        </players>
        <rules>
          <rule id="101" mutable="false">Players take turns proposing and voting on rule changes.</rule>
          <rule id="106" mutable="true">The first player to reach 100 points wins the game.</rule>
          <rule id="107" mutable="true">Players can choose to skip their turn and gain 5 points instead of proposing a rule.</rule>
        </rules>
        <prevturn>
          <proposed_rules>
            <rule player="David">108: If a player's rule is adopted unanimously, they gain an extra 5 points.</rule>
            <rule player="Eve">109: Players can spend 10 points to change the order of turns for the next round.</rule>
          </proposed_rules>
          <votes>
            <vote player="David">Aye, Nay</vote>
            <vote player="Eve">Nay, Aye</vote>
            <vote player="Frank">Aye, Nay</vote>
            <vote player="Grace">Aye, Aye</vote>
          </votes>
        </prevturn>
      </gamestate>

      <gamestate>
        <turn>20</turn>
        <players>
          <player name="Hank" points="150"/>
          <player name="Ivy" points="140"/>
          <player name="Jack" points="145"/>
          <player name="Kate" points="155"/>
        </players>
        <rules>
          <rule id="101" mutable="false">Players take turns proposing and voting on rule changes.</rule>
          <rule id="110" mutable="true">If a player proposes a rule that directly contradicts an existing rule, they lose 10 points.</rule>
          <rule id="111" mutable="true">Players can spend 20 points to make one of their adopted rules immutable for the remainder of the game.</rule>
        </rules>
        <prevturn>
          <proposed_rules>
            <rule player="Hank">112: The game ends after 25 turns, and the player with the most points wins.</rule>
            <rule player="Ivy">113: Players can form alliances and share points, but if one player in the alliance is eliminated, their allied players lose 25 points each.</rule>
          </proposed_rules>
          <votes>
            <vote player="Hank">Aye, Nay</vote>
            <vote player="Ivy">Nay, Aye</vote>
            <vote player="Jack">Aye, Nay</vote>
            <vote player="Kate">Nay, Aye</vote>
          </votes>
        </prevturn>
      </gamestate>
    </syntheticdata>
    """
    return intro_message


def make_api_call(llm_id, game_state, history):
    current_player = game_state["current_player"]
    # Create an un-BERTed save file
    save_file = "game_history.txt"
    with open(save_file, "w") as file:
        file.write(history)

    # Check the length of the un-BERTed save file
    with open(save_file, "r") as file:
        content = file.read()
    if len(content) > 50000:
        history_summary = summarize_history(content)
    else:
        history_summary = content

    if llm_id == current_player:
        # Make a strategic API call for the current player
        prompt = f"<gamestate>{json.dumps(game_state)}</gamestate>"
        prompt += f"<history>{history_summary}</history>"
        # Define the strategy questions
        strategy_questions = [
            "Considering your current position, what rule proposal would best serve your short-term goals and why?",
            "Looking ahead, what rule change would you propose to set yourself up for success in the medium-term, and how does it align with your character's traits?",
            "Thinking strategically, what rule amendment would you suggest to secure your long-term victory, and how does it exploit potential loopholes or weaknesses in the current ruleset?",
        ]
        # Randomly select a strategy question
        selected_question = random.choice(strategy_questions)
        prompt += selected_question
    else:
        # Make a call to propose, debate, or vote on a proposal
        prompt = f"<gamestate>{json.dumps(game_state)}</gamestate>"
        prompt += f"<history>{history_summary}</history>"
        prompt += "Do you have any proposals, debates, or votes for this turn?"

    # Make the API call using the constructed prompt
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4000,
        temperature=0.6,
        system=send_intro_message(llm_id),
        messages=[{"role": "user", "content": prompt}],
    )

    # Access the content directly from the Message object
    response_content = response.content

    return response_content
def extract_proposed_rules(response):
    pattern = r"<rule.*?>(.*?)</rule>"
    return re.findall(pattern, response)

def extract_votes(response):
    pattern = r"<vote.*?>(.*?)</vote>"
    votes = re.findall(pattern, response)
    vote_counts = []
    for vote in votes:
        vote_count = [1 if v.strip().lower() == "aye" else 0 for v in vote.split(",")]
        vote_counts.append(vote_count)
    return vote_counts


def play_game():
    game_state = {
        "players": [{"id": i, "points": 50} for i in range(1, 26)],
        "current_player": 1,
        "rules": {
            "immutable": [
                "All players must always abide by all the rules then in effect, in the form in which they are then in effect.",
                "Initially, rules 1-6 are immutable, and rules 101-108 are mutable.",
                "A rule change is any of the following:\n   a) the proposal of a new mutable rule\n   b) the repeal, or amendment of a mutable rule\n   c) the transmutation of an immutable rule into a mutable rule or vice versa",
                "All rule changes proposed in the proper way shall be voted on. They will be adopted if and only if they receive the required number of votes.",
                "The game ends when a player achieves a winning condition or when further play is impossible. The winner is determined by the rules in effect at that time.",
                "Rule changes that transmute immutable rules into mutable rules may be adopted if and only if the vote is unanimous among the players. Transmutation shall not be implied, but must be stated explicitly in a proposal to take effect."
            ],
            "mutable": [
                "Players shall alternate in turn order, with the human player going first, followed by the LLMs in numerical order. If a player fails to take their turn within 24 hours, their turn is skipped, and the next player in the order takes their turn.",
                "On their turn, a player may propose up to three rule changes and have them voted on sequentially. The effects of any adopted rule changes are then applied before the next player's turn begins.",
                "A rule change is adopted if and only if a majority of the players vote in favor of it.",
                "If a rule change is defeated, the player who proposed it loses 3 points.",
                "A player cannot propose a rule change that directly contradicts an immutable rule.",
                "The game ends when a player reaches 150 points or when a winning condition defined by a rule change is met.",
                "If the game reaches a state where further play is impossible, the player with the most points is declared the winner.",
                "Each player starts with 50 points. Points are earned or lost as described in the rules."
            ],
        },
    "game_over": False,
        "winner": None,
    }

    history = ""

    while not game_state["game_over"]:
        for llm_id in range(1, 26):
            if game_state["game_over"]:
                break

            response_content = make_api_call(llm_id, game_state, history)
            history += response_content

            game_state = update_game_state(game_state, response.content)  # Use the response content

            for player in game_state["players"]:
                if player["points"] >= 150:
                    game_state["game_over"] = True
                    game_state["winner"] = player["id"]
                    break

            if not game_state["game_over"] and is_play_impossible(game_state):
                game_state["game_over"] = True
                game_state["winner"] = determine_winner(game_state)

            save_game_state(game_state)
            save_game_history(history)

            time.sleep(1)

    print("Game Over!")
    if game_state["winner"]:
        print(f"Player {game_state['winner']} wins!")
    else:
        print("The game ended in a state where further play is impossible.")

def update_game_state(game_state, response):
    # Parse the response to extract the proposed rules and votes
    proposed_rules = extract_proposed_rules(response)
    votes = extract_votes(response)

    # Update the game state based on the votes
    for rule, vote_counts in zip(proposed_rules, votes):
        if sum(vote_counts) > len(game_state["players"]) // 2:
            # Rule is adopted
            game_state["rules"]["mutable"].append(rule)
            game_state["players"][game_state["current_player"] - 1]["points"] += 10
        else:
            # Rule is rejected
            game_state["players"][game_state["current_player"] - 1]["points"] -= 5

    # Update the current player
    game_state["current_player"] = (game_state["current_player"] % len(game_state["players"])) + 1

    return game_state

def is_play_impossible(game_state):
    # Check if there are any valid rule changes left to propose
    if not any(is_valid_rule_change(rule, game_state) for rule in game_state["rules"]["mutable"]):
        return True

        # Check if all players have exhausted their turns
    if game_state["current_player"] > len(game_state["players"]):
        return True

        return False

def determine_winner(game_state):
        # Check if any player has reached the winning point threshold
    for player in game_state["players"]:
        if player["points"] >= 150:
            return player["id"]

        # If no player has reached the winning threshold, the player with the most points wins
    return max(game_state["players"], key=lambda player: player["points"])["id"]

def is_valid_rule_change(rule, game_state):
        # Check if the rule contradicts any immutable rules
    for immutable_rule in game_state["rules"]["immutable"]:
        if contradicts(rule, immutable_rule):
            return False

    return True

def contradicts(rule1, rule2):
    # Simple implementation: check if the rules have conflicting keywords
    conflicting_keywords = ["win", "lose", "gain", "lose", "points", "turn", "order"]
    return any(keyword in rule1.lower() and keyword in rule2.lower() for keyword in conflicting_keywords)

def save_game_state(game_state):
    with open("game_state.json", "w") as file:
        json.dump(game_state, file)

def save_game_history(history):
    with open("game_history.txt", "a") as file:
        file.write(history)

play_game()
