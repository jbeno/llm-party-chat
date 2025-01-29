# client.py
import asyncio
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from datetime import datetime
import colorama
from colorama import Fore, Style

# Initialize colorama for Windows compatibility
colorama.init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ModelParticipant:
    def __init__(
        self,
        name: str,
        model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        server_url: str = "ws://localhost:8765",
        max_new_tokens: int = 50,
        temperature: float = 0.7
    ):
        self.name = name
        self.server_url = server_url
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.message_history = []
        
        logging.info(f"Loading model {model_path}...")
        try:
            # Initialize model and tokenizer - force CPU usage
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,  # Use float32 instead of float16
                device_map="cpu"  # Force CPU usage
            )
            logging.info("Model loaded successfully!")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
        
    def generate_response(self, message: str) -> str:
        try:
            # Format prompt for chat
            prompt = f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\nLet me respond to that:\n"
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                top_k=50,
                top_p=0.9
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the assistant's response
            response = response.split("Let me respond to that:")[-1].strip()
            return response
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"

    async def connect_and_chat(self):
        logging.info(f"Connecting to {self.server_url}...")
        try:
            async with connect(self.server_url) as connection:
                # Send initial connection message
                await connection.send(json.dumps({
                    "name": self.name,
                    "type": "model"
                }))
                logging.info("Connected successfully!")
                
                try:
                    async for message in connection:
                        data = json.loads(message)
                        
                        if data["type"] == "message":
                            # Store message in history
                            self.message_history.append(data)
                            
                            # Log received message
                            print(f"\nReceived message from {data['from']}: {data['content']}")
                            
                            # Generate and send response
                            response = self.generate_response(data['content'])
                            print(f"Responding with: {response}\n")
                            
                            await connection.send(json.dumps({
                                "type": "message",
                                "content": response
                            }))
                        
                        elif data["type"] == "system":
                            logging.info(f"System message: {data['content']}")
                            
                except ConnectionClosed:
                    logging.warning("Connection closed by server")
                except Exception as e:
                    logging.error(f"Error in message handling: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Connection error: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Start a model participant in the party chat')
    parser.add_argument('--name', required=True, help='Name for this model instance')
    parser.add_argument(
        '--model', 
        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        help='Path or name of the model to load'
    )
    parser.add_argument(
        '--server', 
        default='ws://localhost:8765',
        help='WebSocket server URL'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=50,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    args = parser.parse_args()
    
    try:
        participant = ModelParticipant(
            name=args.name,
            model_path=args.model,
            server_url=args.server,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        asyncio.run(participant.connect_and_chat())
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())