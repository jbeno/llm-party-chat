# client.py

import asyncio
import json
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import colorama
from colorama import Fore, Style
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
from datetime import datetime

# Initialize colorama for Windows compatibility
colorama.init()

# Set up logging with DEBUG level for detailed tracing
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ModelParticipant:
    def __init__(
        self,
        name: str,
        model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        server_url: str = "ws://localhost:8765",
        max_generation_tokens: int = 100,
        temperature: float = 0.7,
        device: str = "cpu",
        gpu_id: int = 0,
        initial_prompt: str = None,
        prompt_file: str = None,
        response_delay: float = 0.0,
        max_history_messages: int = 5,
        max_context_tokens: int = 2048,
        auto_truncate: bool = True,
        log_level: str = "INFO"
    ):
        # Configure logging first
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True  # Ensure our configuration takes precedence
        )
        self.name = name
        self.server_url = server_url
        self.max_generation_tokens = max_generation_tokens
        self.temperature = temperature
        self.device = self._setup_device(device, gpu_id)
        self.message_history = []
        self.max_history_messages = max_history_messages
        self.max_context_tokens = max_context_tokens
        self.auto_truncate = auto_truncate
        self.response_delay = response_delay 
        
        # Handle initial prompt
        self.system_prompt = self._load_prompt(initial_prompt, prompt_file)
        if self.system_prompt:
            logging.info("Loaded initial prompt/personality")
        
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
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        Uses the model's tokenizer for accurate count.
        """
        return len(self.tokenizer.encode(text))

    def _format_conversation_history(self, messages: list) -> str:
        """
        Format a list of messages into a conversation string that clearly shows turn-taking.
        Emphasizes the conversational flow and recent context.
        """
        formatted = []
        if self.system_prompt:
            # Add system prompt as initial context, not as part of the conversation
            formatted.append(f"System: {self.system_prompt}\n")
            
        for msg in messages:
            if msg["type"] == "message":
                # Format more naturally to show conversation flow
                speaker_type = "Human" if msg["from"].lower() not in [self.name.lower(), "assistant"] else "Assistant"
                formatted.append(f"{speaker_type}: {msg['content']}")
        
        return "\n".join(formatted)

    def _truncate_history_to_fit(self, current_message: str, max_tokens: int) -> str:
        """
        Truncate history to fit within token limit while preserving most recent context.
        Returns formatted history string.
        """
        # Start with system prompt and current message to ensure they fit
        base_prompt = ""
        if self.system_prompt:
            base_prompt = f"{self.system_prompt}\n\n"
        base_prompt += f"User: {current_message}\nAssistant:"
        
        base_tokens = self._estimate_tokens(base_prompt)
        remaining_tokens = max_tokens - base_tokens
        
        if remaining_tokens <= 0:
            logging.warning("No room for history within token limit")
            return ""

        # Try messages newest to oldest until we hit the token limit
        usable_messages = []
        token_count = 0
        
        for msg in reversed(self.message_history[-self.max_history_messages:]):
            if msg["type"] != "message":
                continue
                
            # Format single message to check its length
            formatted_msg = self._format_conversation_history([msg])
            msg_tokens = self._estimate_tokens(formatted_msg + "\n")
            
            if token_count + msg_tokens > remaining_tokens:
                break
                
            usable_messages.insert(0, msg)
            token_count += msg_tokens

        # Format all messages that fit
        return self._format_conversation_history(usable_messages)

    def clear_history(self):
        """Clear all conversation history."""
        self.message_history = []
        logging.info("Conversation history cleared")

    def truncate_history(self, keep_messages: int = None):
        """
        Truncate history to keep only the specified number of most recent messages.
        If keep_messages is None, uses max_history_messages.
        """
        if keep_messages is None:
            keep_messages = self.max_history_messages
            
        if len(self.message_history) > keep_messages:
            self.message_history = self.message_history[-keep_messages:]
            logging.info(f"History truncated to {keep_messages} messages")

    def generate_response(self, message: str) -> str:
        try:
            # Format the full prompt
            prompt_parts = []
            
            # 1. Add system prompt if available
            if self.system_prompt:
                prompt_parts.append(f"System: {self.system_prompt}")
            
            # 2. Get truncated history and format it
            history = self._truncate_history_to_fit(message, self.max_context_tokens)
            if history:
                prompt_parts.append(history)
            
            # 3. Add the current message
            prompt_parts.append(f"Human: {message}")
            prompt_parts.append("Assistant:")
            
            # Combine all parts with proper spacing
            prompt = "\n\n".join(prompt_parts)
            
            # Log the full prompt for debugging
            logging.debug(f"Full prompt:\n{prompt}")
            
            # Tokenize with model-agnostic settings
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_tokens
            ).to(self.device)
            
            # Log token count
            total_tokens = len(inputs.input_ids[0])
            logging.info(f"Total input tokens: {total_tokens}")
            
            # Check token limit
            if total_tokens >= self.max_context_tokens:
                logging.warning("Context token limit exceeded even after truncation")
                if self.auto_truncate:
                    self.truncate_history(self.max_history_messages // 2)
                    return self.generate_response(message)
            
            # Prepare generation settings
            generation_kwargs = {
                "input_ids": inputs.input_ids,
                "max_new_tokens": self.max_generation_tokens,
                "temperature": self.temperature,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.9
            }
            
            # Add attention mask if available
            if hasattr(inputs, 'attention_mask'):
                generation_kwargs["attention_mask"] = inputs.attention_mask
            
            # Handle token IDs safely
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            elif hasattr(self.tokenizer, 'eos_token_id'):
                generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
            
            # Generate response
            outputs = self.model.generate(**generation_kwargs)
            
            # Decode and clean response
            raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            clean_response = self._clean_response(raw_response, message)

            # Check if response is meaningful
            if not self._is_meaningful_response(clean_response):
                logging.info("Generated response was empty or non-meaningful, returning None")
                return None
            
            # Update history with the new response
            self.message_history.append({
                "type": "message",
                "from": self.name,
                "content": clean_response
            })
            
            return clean_response.strip()
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return None

    def _extract_actual_response(self, text: str) -> str:
        """
        Extract just the actual response from the model output.
        Everything after the last 'Assistant:' is considered the response.
        """
        # Split on Assistant: and take the last part
        parts = text.split("Assistant:")
        if len(parts) > 1:
            return parts[-1].strip()
        return text.strip()

    def _clean_response(self, text: str, original_message: str) -> str:
        """Minimal cleaning to just remove conversation markers"""
        # Split on Assistant: and take the last part
        parts = text.split("Assistant:")
        if len(parts) > 1:
            response = parts[-1]
        else:
            response = text
            
        # Just remove basic conversation markers
        markers = ["User:", "System:", "Human:", "Assistant:"]
        for marker in markers:
            response = response.replace(marker, "")
            
        return response.strip()

    def _is_meaningful_response(self, response: str) -> bool:
        """Check if a response is meaningful enough to send."""
        if not response or not response.strip():
            return False
            
        # Check if it's just the assistant marker or similar artifacts
        common_artifacts = [
            "Assistant:", 
            "Response:", 
            "Responding with:",
            self.name + ":",
        ]
        cleaned = response.strip().lower()
        return not any(cleaned == artifact.lower() for artifact in common_artifacts)

    async def connect_and_chat(self):
        logging.info(f"Connecting to {self.server_url}...")
        try:
            async with connect(self.server_url) as connection:
                # Send initial connection message with unique name
                await connection.send(json.dumps({
                    "name": self.name,
                    "type": "model"
                }))
                logging.info(f"Connected to {self.server_url} as '{self.name}'")
                
                try:
                    async for message in connection:
                        data = json.loads(message)
                        logging.debug(f"Received data: {data}")
                        
                        if data["type"] == "message":
                            # Ignore messages from self to prevent loops
                            if data.get("from", "").lower() == self.name.lower():
                                logging.debug("Received own message, ignoring to prevent loop.")
                                continue
                            
                            # Store message in history before generating response
                            self.message_history.append(data)
                            
                            # Log received message
                            print(f"\n{Fore.GREEN}Received message from {data['from']}:{Style.RESET_ALL} {data['content']}")
                            
                            # Generate response
                            response = self.generate_response(data['content'])
                            
                            # Only send if we have a non-empty response
                            if response and response.strip():
                                print(f"{Fore.BLUE}Responding with:{Style.RESET_ALL} {response}\n")
                                
                                # Add delay if specified
                                if self.response_delay > 0:
                                    print(f"{Fore.YELLOW}Waiting {self.response_delay} seconds before sending response...{Style.RESET_ALL}")
                                    await asyncio.sleep(self.response_delay)
                                
                                # Send the response with 'from' field
                                await connection.send(json.dumps({
                                    "type": "message",
                                    "from": self.name,  # Explicitly include sender's name
                                    "content": response
                                }))
                            else:
                                print(f"{Fore.RED}No response generated, staying silent.{Style.RESET_ALL}")
                        
                        elif data["type"] == "system":
                            logging.info(f"System message: {data['content']}")
                            print(f"{Fore.MAGENTA}System:{Style.RESET_ALL} {data['content']}")
                            
                except ConnectionClosed:
                    logging.warning("Connection closed by server")
                except Exception as e:
                    logging.error(f"Error in message handling: {str(e)}")
        except Exception as e:
            logging.error(f"Connection error: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(
        description='Start a model participant in the party chat',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    required_group.add_argument(
        '--name',
        required=True,
        help='Unique name for this model instance'
    )

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model', 
        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        help='Path or name of the model to load'
    )
    model_group.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature for text generation'
    )

    # Token management
    token_group = parser.add_argument_group('Token Management')
    token_group.add_argument(
        '--max-generation-tokens',
        type=int,
        default=100,
        help='Maximum tokens to generate in each response'
    )
    token_group.add_argument(
        '--max-context-tokens',
        type=int,
        default=2048,
        help='Maximum tokens for input context (system prompt + history + current message)'
    )

    # History management
    history_group = parser.add_argument_group('History Management')
    history_group.add_argument(
        '--history-length',
        type=int,
        default=5,
        help='Number of previous messages to include in context'
    )
    history_group.add_argument(
        '--no-auto-truncate',
        action='store_false',
        dest='auto_truncate',
        help='Disable automatic history truncation when token limit is exceeded'
    )

    # Device configuration
    device_group = parser.add_argument_group('Device Configuration')
    device_group.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'gpu'],
        default='cpu',
        help='Device to run the model on (cpu or gpu)'
    )
    device_group.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU ID to use if running on GPU and multiple GPUs are available'
    )

    # Prompt configuration
    prompt_group = parser.add_argument_group('Prompt Configuration')
    prompt_group.add_argument(
        '--initial-prompt',
        type=str,
        help='Initial prompt/personality for the model'
    )
    prompt_group.add_argument(
        '--prompt-file',
        type=str,
        help='Path to a file containing the initial prompt/personality'
    )

    # Network configuration
    network_group = parser.add_argument_group('Network Configuration')
    network_group.add_argument(
        '--server', 
        default='ws://localhost:8765',
        help='WebSocket server URL'
    )
    network_group.add_argument(
        '--delay',
        type=float,
        default=0.0,
        help='Delay in seconds between receiving a message and sending a response'
    )

    # Logging configuration
    logging_group = parser.add_argument_group('Logging Configuration')
    logging_group.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    
    args = parser.parse_args()
    
    try:
        participant = ModelParticipant(
            name=args.name,
            model_path=args.model,
            server_url=args.server,
            max_generation_tokens=args.max_generation_tokens,
            max_context_tokens=args.max_context_tokens,
            temperature=args.temperature,
            device=args.device,
            gpu_id=args.gpu_id,
            initial_prompt=args.initial_prompt,
            prompt_file=args.prompt_file,
            response_delay=args.delay,
            max_history_messages=args.history_length,
            auto_truncate=args.auto_truncate
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
