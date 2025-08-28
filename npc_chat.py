import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

import requests
from flask import Flask, render_template, jsonify, request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('npc_chat.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PlayerMessage:
    """Represents a player message with metadata."""
    player_id: int
    text: str
    timestamp: str
    
    @property
    def datetime_obj(self) -> datetime:
        """Convert timestamp string to datetime object."""
        return datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))

@dataclass
class NPCResponse:
    """Represents an NPC response with context."""
    player_id: int
    player_message: str
    npc_reply: str
    conversation_state: List[str]
    mood: str
    timestamp: str
    processing_time: float

class MoodTracker:
    """Tracks and updates NPC mood based on player interactions."""
    
    MOOD_KEYWORDS = {
        'friendly': ['hello', 'hi', 'thanks', 'thank you', 'help', 'please', 'good', 'great', 'awesome', 'love', 'appreciate', 'welcome', 'friend'],
        'angry': ['useless', 'stupid', 'idiot', 'hate', 'terrible', 'worst', 'annoying', 'blocking', 'move', 'hurry', 'slow', 'boring', 'waste'],
        'neutral': []
    }
    
    @classmethod
    def analyze_mood(cls, text: str, current_mood: str = 'neutral') -> str:
        """Analyze message text and return appropriate mood."""
        text_lower = text.lower()
        
        # Count keywords for each mood
        friendly_count = sum(1 for word in cls.MOOD_KEYWORDS['friendly'] if word in text_lower)
        angry_count = sum(1 for word in cls.MOOD_KEYWORDS['angry'] if word in text_lower)
        
        # Determine mood based on keyword presence and current state
        if angry_count > friendly_count and angry_count > 0:
            return 'angry'
        elif friendly_count > angry_count and friendly_count > 0:
            return 'friendly'
        else:
            # Gradual mood decay towards neutral
            if current_mood == 'angry':
                return 'neutral' if 'sorry' in text_lower else 'angry'
            return current_mood

class ConversationState:
    """Manages conversation state for each player."""
    
    def __init__(self, max_history: int = 3):
        self.max_history = max_history
        self.player_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.player_moods: Dict[int, str] = defaultdict(lambda: 'neutral')
    
    def add_message(self, player_id: int, message: str) -> None:
        """Add a message to player's conversation history."""
        self.player_history[player_id].append(message)
    
    def get_context(self, player_id: int) -> List[str]:
        """Get conversation context for a player."""
        return list(self.player_history[player_id])
    
    def update_mood(self, player_id: int, message: str) -> str:
        """Update and return player's mood based on message."""
        current_mood = self.player_moods[player_id]
        new_mood = MoodTracker.analyze_mood(message, current_mood)
        self.player_moods[player_id] = new_mood
        return new_mood
    
    def get_mood(self, player_id: int) -> str:
        """Get current mood for a player."""
        return self.player_moods[player_id]

class AIProvider:
    """Abstract base class for AI providers."""
    
    def generate_response(self, message: str, context: List[str], mood: str) -> str:
        """Generate NPC response based on message, context, and mood."""
        raise NotImplementedError

class OpenAIProvider(AIProvider):
    """OpenAI API provider for generating NPC responses."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def generate_response(self, message: str, context: List[str], mood: str) -> str:
        """Generate response using OpenAI API."""
        try:
            # Build system prompt based on mood
            mood_prompts = {
                'friendly': "You are a helpful and cheerful NPC who loves to assist players. Respond warmly and enthusiastically.",
                'angry': "You are an irritated NPC who is frustrated but still somewhat helpful. Respond curtly but not rudely.",
                'neutral': "You are a balanced NPC who provides helpful information in a professional manner."
            }
            
            system_prompt = f"{mood_prompts.get(mood, mood_prompts['neutral'])} Keep responses brief (1-2 sentences) and stay in character as a fantasy game NPC."
            
            # Build conversation context
            context_text = ""
            if context:
                context_text = f"\nRecent conversation: {' | '.join(context[-3:])}"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt + context_text},
                    {"role": "user", "content": message}
                ],
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._fallback_response(message, mood)
    
    def _fallback_response(self, message: str, mood: str) -> str:
        """Provide fallback responses when API fails."""
        fallback_responses = {
            'friendly': "Greetings, traveler! How wonderful to meet you!",
            'angry': "I'm quite busy right now, but what do you need?",
            'neutral': "Hello there. How may I assist you today?"
        }
        return fallback_responses.get(mood, fallback_responses['neutral'])

class LocalLLMProvider(AIProvider):
    """Local LLM provider using Ollama or similar local models."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model
    
    def generate_response(self, message: str, context: List[str], mood: str) -> str:
        """Generate response using local LLM."""
        try:
            # Build prompt with context and mood
            mood_instructions = {
                'friendly': "Respond as a cheerful, helpful NPC who enjoys helping players.",
                'angry': "Respond as an irritated NPC who is frustrated but still somewhat professional.",
                'neutral': "Respond as a balanced, professional NPC."
            }
            
            prompt = f"{mood_instructions.get(mood, mood_instructions['neutral'])}\n"
            
            if context:
                prompt += f"Recent conversation: {' | '.join(context[-3:])}\n"
            
            prompt += f"Player says: {message}\nNPC responds:"
            
            # Try Ollama API format first
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                raise Exception(f"Local LLM API returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            return self._fallback_response(message, mood)
    
    def _fallback_response(self, message: str, mood: str) -> str:
        """Provide rule-based fallback responses."""
        message_lower = message.lower()
        
        # Simple rule-based responses based on keywords
        if any(word in message_lower for word in ['hello', 'hi', 'greetings']):
            return "Greetings, traveler! Welcome to our humble village." if mood == 'friendly' else "Hello."
        elif any(word in message_lower for word in ['help', 'quest', 'task']):
            return "I'd be delighted to help you on your adventure!" if mood == 'friendly' else "I can assist you."
        elif any(word in message_lower for word in ['where', 'direction', 'way']):
            return "Ah, seeking directions? The path ahead holds many secrets!" if mood == 'friendly' else "Check your map."
        elif any(word in message_lower for word in ['buy', 'sell', 'trade', 'shop']):
            return "The marketplace has fine wares for adventurers!" if mood == 'friendly' else "Try the shops nearby."
        else:
            responses = {
                'friendly': "That's quite interesting! Tell me more, friend!",
                'angry': "I see. Is there anything else?",
                'neutral': "I understand. How can I help you?"
            }
            return responses.get(mood, responses['neutral'])

class NPCChatProcessor:
    """Main processor for handling NPC chat interactions."""
    
    def __init__(self, ai_provider: AIProvider):
        self.ai_provider = ai_provider
        self.conversation_state = ConversationState()
        self.processed_responses: List[NPCResponse] = []
    
    def load_messages(self, file_path: str) -> List[PlayerMessage]:
        """Load and parse messages from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            messages = [PlayerMessage(**msg) for msg in data]
            # Sort by timestamp (chronological order)
            messages.sort(key=lambda x: x.datetime_obj)
            
            logger.info(f"Loaded {len(messages)} messages from {file_path}")
            return messages
            
        except Exception as e:
            logger.error(f"Error loading messages: {e}")
            raise
    
    def process_message(self, message: PlayerMessage) -> NPCResponse:
        """Process a single message and generate NPC response."""
        start_time = datetime.now()
        
        # Update mood and get conversation context
        mood = self.conversation_state.update_mood(message.player_id, message.text)
        context = self.conversation_state.get_context(message.player_id)
        
        # Generate NPC response
        npc_reply = self.ai_provider.generate_response(message.text, context, mood)
        
        # Add message to conversation history
        self.conversation_state.add_message(message.player_id, message.text)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create response object
        response = NPCResponse(
            player_id=message.player_id,
            player_message=message.text,
            npc_reply=npc_reply,
            conversation_state=context.copy(),
            mood=mood,
            timestamp=message.timestamp,
            processing_time=processing_time
        )
        
        self.processed_responses.append(response)
        
        # Log the interaction
        logger.info(f"Player {message.player_id} [{mood}]: {message.text}")
        logger.info(f"NPC Reply: {npc_reply}")
        logger.info(f"Processing time: {processing_time:.2f}s")
        logger.info("-" * 50)
        
        return response
    
    def process_all_messages(self, messages: List[PlayerMessage]) -> List[NPCResponse]:
        """Process all messages and return responses."""
        logger.info(f"Processing {len(messages)} messages...")
        
        responses = []
        for i, message in enumerate(messages, 1):
            try:
                response = self.process_message(message)
                responses.append(response)
                
                # Progress logging
                if i % 10 == 0:
                    logger.info(f"Processed {i}/{len(messages)} messages")
                    
            except Exception as e:
                logger.error(f"Error processing message {i}: {e}")
                continue
        
        logger.info(f"Completed processing {len(responses)} messages")
        return responses
    
    def export_results(self, output_file: str = 'npc_responses.json') -> None:
        """Export processed results to JSON file."""
        try:
            results = [asdict(response) for response in self.processed_responses]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")

# Flask application
app = Flask(__name__)
processor = None

@app.route('/')
def index():
    """Main page with processing controls and results."""
    return render_template('index.html')

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Configure AI provider settings."""
    if request.method == 'POST':
        config_data = request.json
        provider_type = config_data.get('provider_type', 'local')
        
        global processor
        
        try:
            if provider_type == 'openai':
                api_key = config_data.get('api_key')
                model = config_data.get('model', 'gpt-3.5-turbo')
                if not api_key:
                    return jsonify({'error': 'OpenAI API key required'}), 400
                ai_provider = OpenAIProvider(api_key, model)
            else:  # local
                base_url = config_data.get('base_url', 'http://localhost:11434')
                model = config_data.get('model', 'llama2')
                ai_provider = LocalLLMProvider(base_url, model)
            
            processor = NPCChatProcessor(ai_provider)
            return jsonify({'status': 'success', 'provider': provider_type})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # GET request - return current config
    return jsonify({
        'provider_configured': processor is not None,
        'available_providers': ['openai', 'local']
    })

@app.route('/api/process', methods=['POST'])
def process_messages():
    """Process all messages from players.json."""
    global processor
    
    if not processor:
        return jsonify({'error': 'AI provider not configured'}), 400
    
    try:
        # Load messages
        messages = processor.load_messages('players.json')
        
        # Process messages
        responses = processor.process_all_messages(messages)
        
        # Export results
        processor.export_results()
        
        return jsonify({
            'status': 'success',
            'messages_processed': len(responses),
            'total_messages': len(messages)
        })
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results')
def get_results():
    """Get processed results."""
    global processor
    
    if not processor:
        return jsonify({'error': 'No results available'}), 400
    
    # Return recent results (limit to avoid large responses)
    recent_results = processor.processed_responses[-50:] if len(processor.processed_responses) > 50 else processor.processed_responses
    
    return jsonify([asdict(response) for response in recent_results])

@app.route('/api/stats')
def get_stats():
    """Get processing statistics."""
    global processor
    
    if not processor or not processor.processed_responses:
        return jsonify({'error': 'No statistics available'}), 400
    
    responses = processor.processed_responses
    
    # Calculate statistics
    total_messages = len(responses)
    avg_processing_time = sum(r.processing_time for r in responses) / total_messages
    mood_distribution = {}
    players_count = len(set(r.player_id for r in responses))
    
    # Count mood distribution
    for response in responses:
        mood = response.mood
        mood_distribution[mood] = mood_distribution.get(mood, 0) + 1
    
    return jsonify({
        'total_messages': total_messages,
        'unique_players': players_count,
        'avg_processing_time': round(avg_processing_time, 3),
        'mood_distribution': mood_distribution
    })

def main():
    """Main entry point for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI-Powered NPC Chat System')
    parser.add_argument('--provider', choices=['openai', 'local'], default='local',
                       help='AI provider to use (default: local)')
    parser.add_argument('--openai-key', help='OpenAI API key')
    parser.add_argument('--openai-model', default='gpt-3.5-turbo',
                       help='OpenAI model to use (default: gpt-3.5-turbo)')
    parser.add_argument('--local-url', default='http://localhost:11434',
                       help='Local LLM API URL (default: http://localhost:11434)')
    parser.add_argument('--local-model', default='llama2',
                       help='Local LLM model name (default: llama2)')
    parser.add_argument('--web', action='store_true',
                       help='Run web interface instead of CLI processing')
    
    args = parser.parse_args()
    
    if args.web:
        # Run Flask web interface
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        # Run CLI processing
        try:
            # Configure AI provider
            if args.provider == 'openai':
                if not args.openai_key:
                    api_key = os.getenv('OPENAI_API_KEY')
                    if not api_key:
                        print("Error: OpenAI API key required. Use --openai-key or set OPENAI_API_KEY environment variable.")
                        return
                else:
                    api_key = args.openai_key
                
                ai_provider = OpenAIProvider(api_key, args.openai_model)
                print(f"Using OpenAI API with model: {args.openai_model}")
            else:
                ai_provider = LocalLLMProvider(args.local_url, args.local_model)
                print(f"Using local LLM: {args.local_model} at {args.local_url}")
            
            # Create processor
            processor = NPCChatProcessor(ai_provider)
            
            # Load and process messages
            messages = processor.load_messages('players.json')
            responses = processor.process_all_messages(messages)
            
            # Export results
            processor.export_results()
            
            print(f"\nProcessing complete!")
            print(f"- Processed {len(responses)} messages")
            print(f"- Results saved to npc_responses.json")
            print(f"- Logs saved to npc_chat.log")
            
        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == '__main__':
    main()
