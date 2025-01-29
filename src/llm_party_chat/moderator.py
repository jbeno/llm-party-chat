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

async def moderate():
    print(f"{Fore.WHITE}Connecting to chat server...{Style.RESET_ALL}")
    
    # Using the new connect API
    async with connect("ws://localhost:8765") as connection:
        # Register as a human moderator
        await connection.send(json.dumps({
            "name": "Moderator",
            "type": "human"
        }))
        
        # Handle receiving messages
        async def receive_messages():
            try:
                while True:
                    message = await connection.recv()
                    data = json.loads(message)
                    if data["type"] == "system":
                        print(f"\n{Fore.WHITE}System: {data['content']}{Style.RESET_ALL}")
                    elif data["type"] == "message":
                        # Print incoming messages with sender's name
                        print(f"\n{Fore.CYAN}{data['from']}: {data['content']}{Style.RESET_ALL}")
            except ConnectionClosed:
                print(f"\n{Fore.RED}Disconnected from server{Style.RESET_ALL}")
                
        # Handle sending messages
        async def send_messages():
            try:
                print(f"\n{Fore.WHITE}Enter prompts/questions (press Ctrl+C to exit):{Style.RESET_ALL}")
                while True:
                    content = await ainput(f"{Fore.GREEN}> {Style.RESET_ALL}")
                    if content.strip():
                        await connection.send(json.dumps({
                            "type": "message",
                            "content": content
                        }))
            except KeyboardInterrupt:
                print(f"\n{Fore.WHITE}Exiting...{Style.RESET_ALL}")
                
        # Run both tasks concurrently
        await asyncio.gather(receive_messages(), send_messages())

def main():
    print(f"{Fore.WHITE}=== LLM Party Chat Moderator ==={Style.RESET_ALL}")
    print(f"{Fore.WHITE}This interface allows you to:")
    print("1. Monitor all model conversations")
    print("2. Send prompts to the models")
    print(f"3. See when models join/leave the chat{Style.RESET_ALL}")
    
    try:
        asyncio.run(moderate())
    except KeyboardInterrupt:
        print(f"\n{Fore.WHITE}Goodbye!{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())