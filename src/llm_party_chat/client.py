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
        temperature: float = 0.7,
        device: str = "cpu",
        gpu_id: int = 0,
        initial_prompt: str = None,
        prompt_file: str = None,
        response_delay: float = 0.0
    ):
        self.name = name
        self.server_url = server_url
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.message_history = []
        
        self.response_delay = response_delay
        
        # Handle initial prompt
        self.system_prompt = self._load_prompt(initial_prompt, prompt_file)
        if self.system_prompt:
            logging.info("Loaded initial prompt/personality")
        
        # Handle device selection
        self.device = self._setup_device(device, gpu_id)
        logging.info(f"Using device: {self.device}")
        
        logging.info(f"Loading model {model_path}...")
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Initialize model with appropriate device settings
            model_kwargs = {
                "torch_dtype": torch.float16 if "cuda" in self.device else torch.float32
            }
            
            if "cuda" in self.device:
                model_kwargs["device_map"] = self.device
            else:
                model_kwargs["device_map"] = "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            logging.info("Model loaded successfully!")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def _load_prompt(self, initial_prompt: str = None, prompt_file: str = None) -> str:
        """
        Load the initial prompt either from a string parameter or from a file.
        Returns the prompt string or None if neither is provided.
        """
        if prompt_file:
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                logging.error(f"Error reading prompt file: {str(e)}")
                return None
                
        return initial_prompt

    def _setup_device(self, device: str, gpu_id: int) -> str:
        """
        Setup and validate the device configuration.
        Returns the appropriate device string for model initialization.
        """
        if device.lower() == "cpu":
            return "cpu"
        
        if not torch.cuda.is_available():
            logging.warning("CUDA is not available. Falling back to CPU.")
            return "cpu"
        
        if device.lower() == "gpu":
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                logging.warning("No GPUs found. Falling back to CPU.")
                return "cpu"
            
            if gpu_id >= num_gpus:
                logging.warning(f"GPU {gpu_id} not found. Using GPU 0 instead.")
                return "cuda:0"
            
            return f"cuda:{gpu_id}"
        
        logging.warning(f"Unknown device '{device}'. Falling back to CPU.")
        return "cpu"
        
    def generate_response(self, message: str) -> str:
        try:
            # Format prompt for chat with system prompt if available
            prompt = ""
            if self.system_prompt:
                prompt += f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\nLet me respond to that:\n"
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                top_k=50,
                top_p=0.9
            )
            
            raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response by removing any conversation markers and extra text
            clean_response = self._clean_response(raw_response)
            return clean_response
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def _clean_response(self, text: str) -> str:
        """
        Clean up model output to extract only the actual response.
        Removes conversation markers, system prompts, and user messages.
        """
        # List of markers to split on - add more if needed
        markers = [
            "<|im_start|>user", 
            "<|im_end|>",
            "<|im_start|>assistant",
            "<|im_start|>system",
            "Let me respond to that:",
            "User:",
            "Assistant:",
            "System:"
        ]
        
        # Get the last relevant part of the response
        response = text
        for marker in markers:
            parts = response.split(marker)
            response = parts[-1].strip()
            
        # Remove any remaining conversation artifacts
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip lines that look like conversation markers
            if any(marker.lower() in line.lower() for marker in ["User:", "Assistant:", "System:"]):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()

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
                            
                            # Add delay if specified
                            if self.response_delay > 0:
                                print(f"Waiting {self.response_delay} seconds before sending response...")
                                await asyncio.sleep(self.response_delay)
                            
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
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'gpu'],
        default='cpu',
        help='Device to run the model on (cpu or gpu)'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU ID to use if running on GPU and multiple GPUs are available'
    )
    parser.add_argument(
        '--initial-prompt',
        type=str,
        help='Initial prompt/personality for the model'
    )
    parser.add_argument(
        '--prompt-file',
        type=str,
        help='Path to a file containing the initial prompt/personality'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.0,
        help='Delay in seconds between receiving a message and sending a response'
    )
    
    args = parser.parse_args()
    
    try:
        participant = ModelParticipant(
            name=args.name,
            model_path=args.model,
            server_url=args.server,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=args.device,
            gpu_id=args.gpu_id,
            initial_prompt=args.initial_prompt,
            prompt_file=args.prompt_file,
            response_delay=args.delay
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