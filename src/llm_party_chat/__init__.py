# src/llm_party_chat/__init__.py
"""LLM Party Chat - A real-time chat system for multiple LLMs"""

__version__ = "0.0.1"
__author__ = "Jim Beno"
__email__ = "jim@jimbeno.net"

from . import server
from . import client
from . import moderator

__all__ = ["server", "client", "moderator"]