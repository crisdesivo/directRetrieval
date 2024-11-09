from abc import ABC, abstractmethod
from typing import List, Dict, Union, Generator, AsyncGenerator, AsyncIterator
import requests
import json
import httpx
import asyncio
# from LLamaCPPServer import LlamaCPPServer, AsyncLlamaCPPServer

# define LLMInferface as Union[SyncLLMInterface, AsyncLLMInterface]
LLMInterface = Union["SyncLLMInterface", "AsyncLLMInterface"]

class SyncLLMInterface(ABC):
    @abstractmethod
    def getResponse(self, messages: List[Dict[str,str]], properties: Union[Dict,None], temperature: int = 0, stream: bool = False) -> Union[Dict,str]:
        ...

class AsyncLLMInterface(ABC):
    @abstractmethod
    async def getResponse(self, messages: List[Dict[str,str]], properties: Union[Dict,None], temperature: int = 0, stream: bool = False) -> Union[Union[Dict,str], AsyncIterator]:
        ...