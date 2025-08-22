import React, { useState, useEffect } from 'react';

interface SubTask {
  id: string;
  text: string;
  completed: boolean;
}

interface Task {
  id: string;
  text: string;
  subTasks: SubTask[];
  completed: boolean;
}

export function TodoList() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchTasks = async () => {
    try {
      const response = await fetch('/api/todos');
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      setTasks(data);
    } catch (error) {
      console.error("Failed to fetch tasks:", error);
      setError('Failed to load tasks.');
    }
  };

  useEffect(() => {
    fetchTasks();
    const intervalId = setInterval(fetchTasks, 5000); // Poll every 5 seconds

    return () => clearInterval(intervalId); // Cleanup on component unmount
  }, []);

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="todo-list">
      {tasks.length === 0 && <p>No tasks yet. Start chatting to generate some!</p>}
      <ul>
        {tasks.map(task => (
          <li key={task.id} className="task-item">
            <span className={task.completed ? 'completed' : ''}>{task.text}</span>
            {task.subTasks && task.subTasks.length > 0 && (
              <ul className="subtask-list">
                {task.subTasks.map(subtask => (
                  <li key={subtask.id} className="subtask-item">
                    <span className={subtask.completed ? 'completed' : ''}>{subtask.text}</span>
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
