## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### OpenAI API
Set your API key as an environment variable:
```bash
export OPENAI_API_KEY=your_key_here
```

### Local LLM
1. Install [Ollama](https://ollama.com/)
2. Pull a model: `ollama pull llama2`
3. Start service: `ollama serve`

## Usage

### Web Interface
```bash
python npc_chat.py --web
```
Open http://localhost:5000 in your browser.

### Command Line
```bash
# Using OpenAI
python npc_chat.py --provider openai --openai-key YOUR_API_KEY

# Using local LLM
python npc_chat.py --provider local --local-model model-name
```

## Command Line Options

```
--provider {openai,local}     AI provider (default: local)
--openai-key TEXT            OpenAI API key
--openai-model TEXT          OpenAI model (default: gpt-3.5-turbo)
--local-url TEXT             Local LLM URL (default: http://localhost:11434)
--local-model TEXT           Local model name (default: llama2)
--web                        Run web interface
```

## Features

- Chronological message processing
- Conversation context tracking (last 3 messages per player)
- Dynamic mood system (friendly, angry, neutral)
- Multiple AI backend support
- Web interface for easy management
- Comprehensive logging

## Output Files

- `npc_responses.json`: Processed results with NPC responses
- `npc_chat.log`: Detailed processing logs

## Input Format

The system expects a `players.json` file with player messages in the following format:
```json
{
  "player_id": 1,
  "message": "Hello there!",
  "timestamp": "2025-08-26T15:01:10"
}
```

## Troubleshooting

**OpenAI API Issues:**
- Ensure OPENAI_API_KEY is set
- Check API key validity and credits

**Local LLM Issues:**
- Start Ollama service: `ollama serve`
- Verify model is installed: `ollama list`
- Check URL accessibility: http://localhost:11434

**Performance Issues:**
- Use smaller models for local LLM
- Consider gpt-3.5-turbo for OpenAI
