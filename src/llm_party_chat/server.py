# server.py

import asyncio
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
from websockets.server import ServerConnection
import json
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
import colorama
from colorama import Fore, Style
import logging
import uuid
import argparse
import os
from pathlib import Path

# Initialize colorama for Windows compatibility
colorama.init()

@dataclass
class ModelClient:
    connection: ServerConnection
    name: str
    client_type: str  # 'model' or 'human'
    color: str
    id: str  # Unique identifier

class PartyChat:
    def __init__(self):
        self.clients: Dict[str, ModelClient] = {}
        self.message_history: List[dict] = []
        self.colors = [Fore.BLUE, Fore.GREEN, Fore.YELLOW, Fore.MAGENTA, Fore.CYAN]
        self.color_index = 0
        self._lock = asyncio.Lock()  # Add lock for thread safety

    def get_next_color(self) -> str:
        color = self.colors[self.color_index]
        self.color_index = (self.color_index + 1) % len(self.colors)
        return color

    async def register(self, connection: ServerConnection, client_info: dict) -> str | None:
        """Register a new client. Returns client_id if successful, None if registration fails."""
        async with self._lock:  # Use lock for thread-safe registration
            client_id = str(uuid.uuid4())
            color = self.get_next_color()
            client_name = client_info.get('name', f"Client_{client_id[:8]}")
            client_type = client_info.get('type', 'model')

            # Check for unique name
            if any(client.name == client_name for client in self.clients.values()):
                await connection.send(json.dumps({
                    "type": "system",
                    "content": f"Name '{client_name}' is already taken. Connection rejected."
                }))
                await connection.close()
                logging.warning(f"Connection rejected for duplicate name '{client_name}'")
                return None

            try:
                # Register the client
                self.clients[client_id] = ModelClient(
                    connection=connection,
                    name=client_name,
                    client_type=client_type,
                    color=color,
                    id=client_id
                )

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_message = f"[{timestamp}] {color}{client_name} ({client_type}) joined the chat{Style.RESET_ALL}"
                logging.info(log_message)

                await connection.send(json.dumps({
                    "type": "system",
                    "content": f"Connected as {client_name}"
                }))

                return client_id
            except Exception as e:
                logging.error(f"Error during client registration: {str(e)}")
                return None

    async def unregister(self, connection: ServerConnection) -> None:
        async with self._lock:  # Use lock for thread-safe unregistration
            client_id = None
            for cid, client in self.clients.items():
                if client.connection == connection:
                    client_id = cid
                    break

            if client_id:
                client = self.clients[client_id]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_message = f"[{timestamp}] {client.color}{client.name} ({client.client_type}) left the chat{Style.RESET_ALL}"
                logging.info(log_message)
                del self.clients[client_id]

    async def broadcast(self, message: dict, source_id: str) -> None:
        """Broadcast a message to all clients except the source."""
        if not isinstance(message, dict):
            logging.error(f"Invalid message type: {type(message)}")
            return

        self.message_history.append(message)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source_client = self.clients.get(source_id)

        if source_client:
            if source_client:
                log_message = f"[{timestamp}] {source_client.color}{message.get('from', 'Unknown')} ({source_client.client_type}): {message.get('content', '')}{Style.RESET_ALL}"
                logging.info(log_message)

        # Create a safe copy of clients for iteration
        async with self._lock:
            clients_to_broadcast = list(self.clients.items())

        # Broadcast to all clients except the source
        for client_id, client in clients_to_broadcast:
            if client_id != source_id:
                try:
                    await client.connection.send(json.dumps(message))
                    logging.debug(f"Message sent to '{client.name}'")
                except ConnectionClosed:
                    logging.warning(f"Connection closed for client '{client.name}' during broadcast")
                    asyncio.create_task(self.unregister(client.connection))
                except Exception as e:
                    logging.error(f"Error broadcasting to '{client.name}': {str(e)}")

# Create a single party instance for all connections
party = PartyChat()

async def handler(connection: ServerConnection) -> None:
    """Handle individual client connections."""
    client_id = None
    try:
        # Receive the initial registration message
        message = await connection.recv()
        try:
            client_info = json.loads(message)
        except json.JSONDecodeError:
            await connection.send(json.dumps({
                "type": "system",
                "content": "Invalid JSON in registration message."
            }))
            await connection.close()
            logging.warning("Connection closed due to invalid JSON in registration")
            return

        # Validate client_info
        if not all(key in client_info for key in ('name', 'type')):
            await connection.send(json.dumps({
                "type": "system",
                "content": "Invalid registration message. 'name' and 'type' are required."
            }))
            await connection.close()
            logging.warning("Connection closed due to missing required fields in registration")
            return

        client_id = await party.register(connection, client_info)
        if not client_id:
            return  # Registration failed

        async for message in connection:
            try:
                data = json.loads(message)

                # Validate message structure
                if not all(key in data for key in ('type', 'content')):
                    await connection.send(json.dumps({
                        "type": "system",
                        "content": "Invalid message format. 'type' and 'content' are required."
                    }))
                    logging.warning(f"Invalid message format from '{party.clients[client_id].name}'")
                    continue

                # Ensure 'from' field is present
                if 'from' not in data:
                    data['from'] = party.clients[client_id].name

                await party.broadcast(data, client_id)

            except json.JSONDecodeError:
                logging.error(f"Invalid JSON message received from '{party.clients[client_id].name}'")
            except Exception as e:
                logging.error(f"Error handling message from '{party.clients[client_id].name}': {str(e)}")

    except ConnectionClosed:
        if client_id:
            await party.unregister(connection)
    except Exception as e:
        logging.error(f"Handler error: {str(e)}")
        if client_id:
            await party.unregister(connection)

def setup_logging(log_level: str, log_file: str | None = None) -> None:
    """Configure logging with both console and file handlers if specified."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    class ColorStripper(logging.Formatter):
        """Strip color codes when writing to file"""
        def format(self, record):
            # Save original message
            original_msg = record.msg
            # Strip color codes for file
            if isinstance(record.msg, str):
                record.msg = str(record.msg).replace(Fore.WHITE, '').replace(Fore.GREEN, '')\
                    .replace(Fore.BLUE, '').replace(Fore.YELLOW, '').replace(Fore.MAGENTA, '')\
                    .replace(Fore.CYAN, '').replace(Style.RESET_ALL, '')
            # Format the message
            formatted = super().format(record)
            # Restore original message
            record.msg = original_msg
            return formatted

    # Setup handlers
    handlers = []
    
    # Console handler - keep colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    handlers.append(console_handler)

    # File handler - strip colors
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(ColorStripper('%(asctime)s - %(levelname)s - %(message)s'))
        handlers.append(file_handler)

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True
    )

async def main(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Main server function with configurable logging level and file output."""
    # Configure logging
    setup_logging(log_level, log_file)

    server_host = "localhost"
    server_port = 8765
    
    startup_message = f"Starting server on ws://{server_host}:{server_port}"
    logging.info(f"{Fore.WHITE}{startup_message}{Style.RESET_ALL}")
    
    if log_file:
        log_notice = f"Logging to file: {log_file}"
        logging.info(f"{Fore.WHITE}{log_notice}{Style.RESET_ALL}")
    
    logging.info(f"{Fore.WHITE}Log level: {log_level}{Style.RESET_ALL}")
    logging.info(f"{Fore.WHITE}Waiting for connections...{Style.RESET_ALL}")
    logging.info(f"{Fore.WHITE}(Use {Fore.GREEN}python moderator.py{Fore.WHITE} to join as a human moderator){Style.RESET_ALL}")

    try:
        async with serve(handler, server_host, server_port):
            await asyncio.Future()  # Run forever
    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM Party Chat Server')
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file (if not specified, logs only go to console)'
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.log_level, args.log_file))
    except KeyboardInterrupt:
        logging.info("Server shutdown initiated by KeyboardInterrupt")
        print("\nServer has been shut down gracefully")
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
        exit(1)