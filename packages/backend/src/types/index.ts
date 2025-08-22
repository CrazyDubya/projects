export interface SubTask {
  id: string;
  text: string;
  completed: boolean;
}

export interface Task {
  id: string;
  text: string;
  subTasks: SubTask[];
  completed: boolean;
  createdAt: Date;
}

export interface Message {
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
}

export interface Conversation {
  id: string;
  messages: Message[];
}

// For now, the PotentialTaskPool will be a simple array of strings.
// The refinement process will turn these into structured Tasks.
export type PotentialTaskPool = string[];
