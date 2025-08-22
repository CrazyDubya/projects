import fetch from 'node-fetch';

const OLLAMA_API_URL = 'http://localhost';

interface OllamaResponse {
  response: string;
  done: boolean;
}

export async function queryOllama(model: string, prompt: string, port: number): Promise<string> {
  try {
    const response = await fetch(`${OLLAMA_API_URL}:${port}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        prompt,
        stream: false, // For now, we'll get the full response at once
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API request failed with status ${response.status}`);
    }

    const data = await response.json() as OllamaResponse;
    return data.response;
  } catch (error) {
    console.error('Error querying Ollama:', error);
    throw error;
  }
}
