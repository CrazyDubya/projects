import { Router } from 'express';
import { queryOllama } from '../services/ollamaService';
import { Task } from '../types';

const router = Router();

// --- In-memory data stores (for now) ---
const todos: Task[] = [];
const potentialTaskPool: string[] = [];
const onboardingState: { [userId: string]: { step: number; answers: string[] } } = {};

// --- Ollama Configuration ---
const MAIN_OLLAMA_PORT = 11434;
const MAIN_OLLAMA_MODEL = 'llama2';
const SECONDARY_OLLAMA_PORT = 11435; // For refinement
const SECONDARY_OLLAMA_MODEL = 'phi'; // A smaller model

// --- Onboarding Logic ---
const onboardingQuestions = [
  "What are your main goals for using this bot? (e.g., personal productivity, project management)",
  "What kind of tasks do you usually manage? (e.g., work, personal, creative projects)",
  "How do you prefer to structure your to-do list? (e.g., by priority, by project, by due date)",
];

router.post('/onboarding', async (req, res) => {
  const { userId, answer } = req.body;
  if (!userId) {
    return res.status(400).json({ error: 'userId is required' });
  }

  if (!onboardingState[userId]) {
    onboardingState[userId] = { step: 0, answers: [] };
  }

  const state = onboardingState[userId];

  if (answer) {
    state.answers.push(answer);
  }

  if (state.step < onboardingQuestions.length) {
    const question = onboardingQuestions[state.step];
    state.step++;
    res.json({ question });
  } else {
    // Onboarding finished, now the LLM could ask adaptive questions
    // For now, we'll just send a completion message.
    const summary = `Onboarding complete. Your goals: ${state.answers.join('. ')}`;
    try {
      const adaptiveQuestion = await queryOllama(
        MAIN_OLLAMA_MODEL,
        `Based on this summary: "${summary}", ask one clarifying question to help set up the user's to-do list.`,
        MAIN_OLLAMA_PORT
      );
      // In a real app, we'd continue the conversation. For now, we just show it.
      res.json({
        message: "Onboarding complete! Here is an adaptive question the bot could ask:",
        adaptiveQuestion,
      });
    } catch (error) {
      res.status(500).json({ error: "Onboarding completed, but failed to generate adaptive question." });
    }
  }
});


// --- Chat and To-Do Logic ---

// Function to parse #todo hashtags
function parseTodosFromMessage(message: string): string[] {
  const regex = /#todo\s+([^\n#]+)/g;
  const matches = message.match(regex) || [];
  return matches.map(match => match.replace(/#todo\s+/, '').trim());
}

router.post('/chat', async (req, res) => {
  const { message } = req.body;
  if (!message) {
    return res.status(400).json({ error: 'Message is required' });
  }

  // 1. Manually add tasks from #todo
  const manualTasks = parseTodosFromMessage(message);
  if (manualTasks.length > 0) {
    // In a real app, these would be added to the main `todos` list after processing.
    // For now, we'll add them to the potential pool.
    potentialTaskPool.push(...manualTasks);
    console.log(`Added ${manualTasks.length} manual tasks to the pool.`);
  }

  // 2. Get a conversational reply from the main LLM
  try {
    const reply = await queryOllama(MAIN_OLLAMA_MODEL, message, MAIN_OLLAMA_PORT);

    // 3. (Async) Extract potential tasks from the conversation
    queryOllama(
      MAIN_OLLAMA_MODEL,
      `Given the conversation: USER: "${message}" BOT: "${reply}", list any potential tasks for the user as a comma-separated list. If none, say "NONE".`,
      MAIN_OLLAMA_PORT
    ).then(taskResponse => {
      if (taskResponse.trim().toUpperCase() !== 'NONE') {
        const extractedTasks = taskResponse.split(',').map(t => t.trim());
        potentialTaskPool.push(...extractedTasks);
        console.log(`Added ${extractedTasks.length} extracted tasks to the pool.`);
      }
    });

    res.json({ reply });
  } catch (error) {
    res.status(500).json({ error: 'Failed to get response from Ollama' });
  }
});

// --- To-Do CRUD ---
router.get('/todos', (req, res) => {
  res.json(todos);
});

router.post('/todos', (req, res) => {
  const { text } = req.body;
  if (!text) {
    return res.status(400).json({ error: 'Task text is required' });
  }
  const newTask: Task = {
    id: Date.now().toString(),
    text,
    subTasks: [],
    completed: false,
    createdAt: new Date(),
  };
  todos.push(newTask);
  res.status(201).json(newTask);
});

// Placeholder for the background refinement process
setInterval(() => {
  if (potentialTaskPool.length > 0) {
    const taskToRefine = potentialTaskPool.shift()!;
    console.log(`Refining task: "${taskToRefine}"`);
    // Use the secondary, smaller model for refinement
    queryOllama(
      SECONDARY_OLLAMA_MODEL,
      `Given the task: "${taskToRefine}", break it down into a main task and a few sub-tasks. Format as JSON: {"text": "Main task", "subTasks": ["sub-task 1", "sub-task 2"]}`,
      SECONDARY_OLLAMA_PORT
    ).then(refinedTaskJson => {
      try {
        const refinedTask = JSON.parse(refinedTaskJson);
        const newTask: Task = {
          id: Date.now().toString(),
          text: refinedTask.text,
          subTasks: refinedTask.subTasks.map((st: string) => ({ id: Date.now().toString(), text: st, completed: false })),
          completed: false,
          createdAt: new Date(),
        };
        todos.push(newTask);
        console.log(`Added refined task to the to-do list: "${newTask.text}"`);
      } catch (e) {
        console.error("Failed to parse refined task JSON:", refinedTaskJson);
        // Put it back in the pool to try again later
        potentialTaskPool.push(taskToRefine);
      }
    });
  }
}, 10000); // Run every 10 seconds


export default router;
