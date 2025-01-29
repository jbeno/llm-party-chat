# server.py
import asyncio
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
import json
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
import colorama
from colorama import Fore, Style

# Initialize colorama for Windows compatibility
colorama.init()

@dataclass
class ModelClient:
    connection: 'ServerConnection'
    name: str
    client_type: str  # 'model' or 'human'
    color: str

class PartyChat:
    def __init__(self):
        self.clients: Dict[str, ModelClient] = {}
        self.message_history: List[dict] = []
        self.colors = [Fore.BLUE, Fore.GREEN, Fore.YELLOW, Fore.MAGENTA, Fore.CYAN]
        self.color_index = 0
    
    def get_next_color(self):
        color = self.colors[self.color_index]
        self.color_index = (self.color_index + 1) % len(self.colors)
        return color

    async def register(self, connection, client_info: dict):
        client_id = str(id(connection))
        color = self.get_next_color()
        self.clients[client_id] = ModelClient(
            connection=connection,
            name=client_info['name'],
            client_type=client_info.get('type', 'model'),
            color=color
        )
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{Fore.WHITE}[{timestamp}] {color}{client_info['name']} joined the chat{Style.RESET_ALL}")
        
        await connection.send(json.dumps({
            "type": "system",
            "content": f"Connected as {client_info['name']}"
        }))

    async def unregister(self, connection):
        client_id = str(id(connection))
        if client_id in self.clients:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{Fore.WHITE}[{timestamp}] {self.clients[client_id].color}{self.clients[client_id].name} left the chat{Style.RESET_ALL}")
            del self.clients[client_id]

    async def broadcast(self, message: dict, source_id: str):
        self.message_history.append(message)
        timestamp = datetime.now().strftime("%H:%M:%S")
        source_client = self.clients.get(source_id)
        
        if source_client:
            print(f"{Fore.WHITE}[{timestamp}] {source_client.color}{message['from']}: {message['content']}{Style.RESET_ALL}")
        
        # Create a safe copy of clients for iteration
        clients_to_broadcast = list(self.clients.items())
        
        # Broadcast to all clients except the source
        for client_id, client in clients_to_broadcast:
            if client_id != source_id:
                try:
                    await client.connection.send(json.dumps(message))
                except ConnectionClosed:
                    # Handle disconnection in a separate task to avoid modifying 
                    # the dictionary during iteration
                    asyncio.create_task(self.unregister(client.connection))
                except Exception as e:
                    print(f"Error broadcasting to {client.name}: {str(e)}")

# Create a single party instance for all connections
party = PartyChat()

async def handler(connection):
    try:
        message = await connection.recv()
        client_info = json.loads(message)
        await party.register(connection, client_info)
        
        async for message in connection:
            try:
                data = json.loads(message)
                client_id = str(id(connection))
                # Add sender name to the message if not already present
                if 'from' not in data:
                    data['from'] = party.clients[client_id].name
                await party.broadcast(data, client_id)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON message received")
            except Exception as e:
                print(f"Error handling message: {str(e)}")
            
    except ConnectionClosed:
        await party.unregister(connection)
    except Exception as e:
        print(f"Handler error: {str(e)}")
        await party.unregister(connection)

async def main():
    print(f"{Fore.WHITE}Starting server on ws://localhost:8765")
    print("Waiting for connections...")
    print(f"(Use {Fore.GREEN}python moderator.py{Fore.WHITE} to join as a human moderator){Style.RESET_ALL}")
    
    async with serve(handler, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())