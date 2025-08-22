import './App.css';
import { ChatWindow } from './components/ChatWindow';
import './components/ChatWindow.css';
import { TodoList } from './components/TodoList';
import './components/TodoList.css';


function App() {
  return (
    <div className="app-container">
      <header>
        <h1>AI To-Do Bot</h1>
      </header>
      <main className="main-content">
        <div className="chat-section">
          <ChatWindow />
        </div>
        <div className="todo-section">
          <h2>To-Do List</h2>
          <TodoList />
        </div>
      </main>
    </div>
  )
}

export default App;
