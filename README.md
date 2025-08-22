# AI To-Do Bot

This project is a simple AI-powered to-do list bot. It uses a chat interface to interact with a user, and based on the conversation, it automatically generates and refines a to-do list.

This repository contains the initial scaffolding for the application. See the "Future Work" section for more details on planned features.

## Project Structure

This is a monorepo using npm workspaces. The code is organized into two main packages:

-   `packages/backend`: An Express.js server that handles the application logic, communicates with the Ollama models, and serves the API.
-   `packages/frontend`: A React application (built with Vite) that provides the user interface for the chat and to-do list.

## Prerequisites

1.  **Node.js and npm**: Make sure you have Node.js (v18 or later) and npm installed.
2.  **Ollama**: You need to have Ollama installed and running. You can find installation instructions on the [Ollama website](https://ollama.ai/).
3.  **AI Models**: You need to have at least two models downloaded. We recommend a larger model for the main chat and a smaller, faster model for refinement.
    ```bash
    ollama pull llama2  # For the main chat
    ollama pull phi     # For task refinement
    ```

## Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies**:
    From the root of the project, run:
    ```bash
    npm install
    ```
    This will install dependencies for both the `backend` and `frontend` packages.

## Running the Application

You will need to run three separate processes in different terminals: the two Ollama model servers and the application itself.

1.  **Run the Main Ollama Model**:
    Open a terminal and run the main model (e.g., `llama2`) on the default port `11434`:
    ```bash
    ollama serve
    ```
    Wait for the model to load. You can open another terminal and run `ollama run llama2` to confirm it's working.

2.  **Run the Secondary Ollama Model**:
    Open another terminal and run the secondary model (e.g., `phi`) on a different port, for example `11435`:
    ```bash
    OLLAMA_HOST=127.0.0.1:11435 ollama serve
    ```
    Wait for the model to load.

3.  **Run the Backend Server**:
    In another terminal, navigate to the root of the project and run:
    ```bash
    npm run dev --workspace=@ai-todo-bot/backend
    ```
    The backend server will start on `http://localhost:3000`.

4.  **Run the Frontend Application**:
    In yet another terminal, navigate to the root of the project and run:
    ```bash
    npm run dev --workspace=@ai-todo-bot/frontend
    ```
    The frontend development server will start, usually on `http://localhost:5173`. Open this URL in your browser to use the application.

## How It Works

-   **Onboarding**: The bot will eventually have a full onboarding flow. For now, the backend has placeholder logic.
-   **Chat**: Interact with the bot through the chat interface.
-   **Manual Todos**: Add a to-do item manually by typing `#todo` followed by your task (e.g., `#todo Write a report`).
-   **Automatic Todos**: The main LLM analyzes your conversation and suggests tasks, which are added to a "potential tasks" pool.
-   **Task Refinement**: A background process uses a smaller, secondary LLM to take tasks from the potential pool, break them down into a main task and sub-tasks, and add them to your official to-do list. The list on the right will update automatically.

## Future Work

This initial implementation provides the core scaffolding. Future work could include:
-   **Full Onboarding**: Implementing the adaptive Q&A logic.
-   **"Hostile but Compliant" Personality**: Adding the logic to detect subversion and have the bot react as specified.
-   **AI-Generated UI/UX**: Allowing the model to influence or generate parts of the user interface.
-   **Persistent Storage**: Replacing the in-memory data stores with a database.
-   **Testing**: Adding a full suite of unit and integration tests.
