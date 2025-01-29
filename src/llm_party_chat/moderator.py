# moderator.py
import asyncio
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
import json
import argparse
from aioconsole import ainput
import colorama
from colorama import Fore, Style

# Initialize colorama for Windows compatibility
colorama.init()

class Moderator:
    def __init__(self, show_messages: bool = True):
        self.show_messages = show_messages

    async def receive_messages(self, connection):
        try:
            while True:
                message = await connection.recv()
                data = json.loads(message)
                if self.show_messages:  # Only show messages if enabled
                    if data["type"] == "system":
                        print(f"\n{Fore.WHITE}System: {data['content']}{Style.RESET_ALL}")
                    elif data["type"] == "message":
                        print(f"\n{Fore.CYAN}{data['from']}: {data['content']}{Style.RESET_ALL}")
        except ConnectionClosed:
            print(f"\n{Fore.RED}Disconnected from server{Style.RESET_ALL}")

    async def send_messages(self, connection):
        try:
            mode_str = "showing" if self.show_messages else "hiding"
            print(f"\n{Fore.WHITE}Enter prompts/questions (press Ctrl+C to exit)")
            print(f"Currently {mode_str} incoming messages")
            print(f"Monitor full conversation in server window{Style.RESET_ALL}")
            
            while True:
                content = await ainput(f"{Fore.GREEN}> {Style.RESET_ALL}")
                if content.strip():
                    if content.lower() == "/toggle":
                        self.show_messages = not self.show_messages
                        mode_str = "showing" if self.show_messages else "hiding"
                        print(f"{Fore.WHITE}Now {mode_str} incoming messages{Style.RESET_ALL}")
                    else:
                        await connection.send(json.dumps({
                            "type": "message",
                            "content": content
                        }))
        except KeyboardInterrupt:
            print(f"\n{Fore.WHITE}Exiting...{Style.RESET_ALL}")

    async def moderate(self):
        print(f"{Fore.WHITE}Connecting to chat server...{Style.RESET_ALL}")
        
        async with connect("ws://localhost:8765") as connection:
            # Register as a human moderator
            await connection.send(json.dumps({
                "name": "Moderator",
                "type": "human"
            }))
            
            # Run both tasks concurrently
            await asyncio.gather(
                self.receive_messages(connection),
                self.send_messages(connection)
            )

def main():
    parser = argparse.ArgumentParser(description='LLM Party Chat Moderator Interface')
    parser.add_argument(
        '--hide-messages',
        action='store_true',
        help='Start with incoming messages hidden'
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
        moderator = Moderator(show_messages=not args.hide_messages)
        asyncio.run(moderator.moderate())
    except KeyboardInterrupt:
        print(f"\n{Fore.WHITE}Goodbye!{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())