# Simple Chatbot Example

A simple command-line chatbot that uses OpenAI and can perform web searches.

## Features

- Uses OpenAI's GPT-4o-mini model for cost-effective conversations
- Supports web search via function calling
- Reads from stdin and writes to stdout
- Loads environment variables from `aiqa-client-python/.env`

## Setup

1. Install dependencies:
   ```bash
   pip install -r ../requirements.examples.txt
   ```

2. Create or update `aiqa-client-python/.env` with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

Run the chatbot:
```bash
python chatbot.py
```

The chatbot will:
- Read your input from stdin
- Respond with AI-generated answers
- Automatically use web search when needed (via OpenAI function calling)
- Exit when you type "exit" or "quit"

## Example

```
You: What is the capital of France?
Bot: The capital of France is Paris.

You: What's the weather like today?
Bot: [Uses web_search tool] Based on current information...

You: exit
Goodbye!
```

## Web Search

The chatbot uses DuckDuckGo's instant answer API for web searches. The OpenAI model automatically decides when to use the web search tool based on the user's question.

