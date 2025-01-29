# LLM Party Chat

A real-time chat system that enables multiple Large Language Models to engage in conversations with each other and human moderators. Models can run on different machines and communicate through a central websocket server.

## Features

- Multi-model conversation support
- Real-time websocket communication
- Human moderation interface
- Color-coded messages for different participants
- Support for any Hugging Face transformers model
- Distributed architecture - models can run on different machines
- Graceful handling of connections/disconnections

## Installation

### Method 1: Quick Setup (Recommended for trying it out)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-party-chat.git
cd llm-party-chat
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install websockets transformers torch colorama aioconsole
```

### Method 2: Install as Package

```bash
pip install llm-party-chat
```

## Usage

### Method 1: Direct Usage (Recommended for development)

1. Start the server:
```bash
python server.py
```

2. In separate terminals, start two or more model clients:
```bash
python client.py --name "Model1"
python client.py --name "Model2"
```

3. Start the moderator interface:
```bash
python moderator.py
```

### Method 2: Package Usage

If you installed via pip:

1. Start the server:
```bash
python -m llm_party_chat.server
```

2. In separate terminals, start model clients:
```bash
python -m llm_party_chat.client --name "Model1"
python -m llm_party_chat.client --name "Model2"
```

3. Start the moderator:
```bash
python -m llm_party_chat.moderator
```

## Components

### Server (`server.py`)
- Central websocket server that manages connections
- Handles message broadcasting
- Manages client registration/disconnection
- Maintains chat history
- Color codes different participants

### Client (`client.py`)
- Loads and runs a language model
- Connects to the server
- Processes incoming messages
- Generates responses using the model
- Supports various model configurations

### Moderator (`moderator.py`)
- Human interface to the chat
- Sends prompts to models
- Monitors all conversations
- Views system status and connections

## Configuration

### Client Configuration
```bash
python client.py \
    --name "Model1" \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --max-tokens 50 \
    --temperature 0.7 \
    --server "ws://localhost:8765"
```

### Server Configuration
```bash
python server.py --host "localhost" --port 8765
```

## Requirements

- Python 3.10+
- websockets
- transformers
- torch
- colorama
- aioconsole

## Development

The repository structure:
```
llm-party-chat/
├── LICENSE
├── MANIFEST.in
├── README.md
├── requirements.txt
├── setup.py
└── src/
    └── llm_party_chat/
        ├── __init__.py
        ├── server.py
        ├── client.py
        └── moderator.py
```

For development:
1. Clone the repository
2. Create a virtual environment
3. Install requirements
4. Run the components directly using Method 1 above

## Future Improvements

- Message persistence
- Web interface
- More model options
- Chat history export
- Authentication
- Docker support

## License

MIT License

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make changes
4. Commit (`git commit -am 'Add feature'`)
5. Push (`git push origin feature/improvement`)
6. Create Pull Request