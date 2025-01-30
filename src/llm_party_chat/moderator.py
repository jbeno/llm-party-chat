# moderator.py

import asyncio
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
import json
import argparse
from aioconsole import ainput
import colorama
from colorama import Fore, Style
import logging

# Initialize colorama for Windows compatibility
colorama.init()

class Moderator:
    def __init__(self, show_messages: bool = True, log_level: str = "INFO"):
        # Configure logging
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True  # Ensure our configuration takes precedence
        )
        
        self.show_messages = show_messages
        logging.info(f"Moderator initialized with message display: {show_messages}")

    async def receive_messages(self, connection):
        try:
            while True:
                message = await connection.recv()
                data = json.loads(message)
                logging.debug(f"Received message: {data}")
                
                if self.show_messages:  # Only show messages if enabled
                    if data["type"] == "system":
                        message_text = f"\n{Fore.WHITE}System: {data['content']}{Style.RESET_ALL}"
                        print(message_text)
                        logging.debug(f"Displayed system message: {data['content']}")
                    elif data["type"] == "message":
                        message_text = f"\n{Fore.CYAN}{data['from']}: {data['content']}{Style.RESET_ALL}"
                        print(message_text)
                        logging.debug(f"Displayed chat message from {data['from']}")
        except ConnectionClosed:
            error_msg = "Disconnected from server"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            logging.error(error_msg)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON received: {str(e)}")
        except Exception as e:
            logging.error(f"Error in receive_messages: {str(e)}")

    async def send_messages(self, connection):
        try:
            mode_str = "showing" if self.show_messages else "hiding"
            print(f"\n{Fore.WHITE}Enter prompts/questions (press Ctrl+C to exit)")
            print(f"Currently {mode_str} incoming messages")
            print(f"Monitor full conversation in server window{Style.RESET_ALL}")
            logging.info("Message input interface initialized")
            
            while True:
                content = await ainput(f"{Fore.GREEN}> {Style.RESET_ALL}")
                if content.strip():
                    if content.lower() == "/toggle":
                        self.show_messages = not self.show_messages
                        mode_str = "showing" if self.show_messages else "hiding"
                        status_msg = f"Now {mode_str} incoming messages"
                        print(f"{Fore.WHITE}{status_msg}{Style.RESET_ALL}")
                        logging.info(status_msg)
                    else:
                        message = {
                            "type": "message",
                            "content": content
                        }
                        await connection.send(json.dumps(message))
                        logging.debug(f"Sent message: {content}")
                        
        except KeyboardInterrupt:
            exit_msg = "Exiting..."
            print(f"\n{Fore.WHITE}{exit_msg}{Style.RESET_ALL}")
            logging.info(exit_msg)
        except Exception as e:
            logging.error(f"Error in send_messages: {str(e)}")

    async def moderate(self):
        server_url = "ws://localhost:8765"
        logging.info(f"Connecting to chat server at {server_url}")
        print(f"{Fore.WHITE}Connecting to chat server...{Style.RESET_ALL}")
        
        try:
            async with connect(server_url) as connection:
                # Register as a human moderator
                registration = {
                    "name": "Moderator",
                    "type": "human"
                }
                await connection.send(json.dumps(registration))
                logging.info("Registered as human moderator")
                
                # Run both tasks concurrently
                await asyncio.gather(
                    self.receive_messages(connection),
                    self.send_messages(connection)
                )
        except ConnectionRefusedError:
            error_msg = "Failed to connect to server. Is the server running?"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            logging.error(error_msg)
        except Exception as e:
            logging.error(f"Error in moderate: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description='LLM Party Chat Moderator Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Interface configuration
    interface_group = parser.add_argument_group('Interface Configuration')
    interface_group.add_argument(
        '--hide-messages',
        action='store_true',
        help='Start with incoming messages hidden'
    )
    
    # Logging configuration
    logging_group = parser.add_argument_group('Logging Configuration')
    logging_group.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
        help='Set the logging level'
    )
    
    args = parser.parse_args()
    
    print(f"{Fore.WHITE}=== LLM Party Chat Moderator ==={Style.RESET_ALL}")
    print(f"{Fore.WHITE}This interface allows you to:")
    print("1. Monitor all model conversations (optional)")
    print("2. Send prompts to the models")
    print("3. See when models join/leave the chat")
    print("\nCommands:")
    print("/toggle - Toggle showing/hiding incoming messages")
    print(f"Ctrl+C  - Exit{Style.RESET_ALL}")
    
    try:
        moderator = Moderator(
            show_messages=not args.hide_messages,
            log_level=args.log_level
        )
        asyncio.run(moderator.moderate())
    except KeyboardInterrupt:
        print(f"\n{Fore.WHITE}Goodbye!{Style.RESET_ALL}")
        logging.info("Moderator shutdown initiated by user")
    except Exception as e:
        error_msg = f"Fatal error: {str(e)}"
        print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logging.critical(error_msg)
        return 1
    return 0

if __name__ == "__main__":
    exit(main())