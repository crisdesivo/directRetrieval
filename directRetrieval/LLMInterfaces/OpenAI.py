from typing import Dict, List, Union, Generator, AsyncGenerator
import requests
import json
import httpx
from ._LLMInterfaces import SyncLLMInterface, AsyncLLMInterface

class OpenAI(AsyncLLMInterface):
    url: str = "https://api.openai.com/v1/chat/completions"
    api_key: str
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def getResponse(
                    self,messages: List[Dict[str,str]],
                    properties: Union[Dict,None],
                    temperature: int = 0,
                    stream: bool = False
                    ) -> Union[Union[Dict,str],Generator[str,None,None]]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data: Dict
        if properties is not None:
            # print(properties)
            data = {
                "model": "gpt-4o-mini",
                'messages': messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "QnA",
                        "schema": {
                            "type": "object",
                            "properties": properties,
                            "required": list(properties.keys()),
                        }
                    }
                },
                'temperature': temperature,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                                        self.url,
                                        headers=headers,
                                        json=data,
                                        timeout=1000000
                                        )
                response = response.json()['choices'][0]['message']['content']
                try:
                    jsonOutput = json.loads(response)
                except Exception as e:
                    print(response)
                    raise e
                return jsonOutput
        else:
            data = {
                "model": "gpt-4o-mini",
                'messages': messages,
                'temperature': temperature,
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                                        self.url,
                                        headers=headers,
                                        json=data,
                                        timeout=1000000,
                                        )
                response = response.json()['choices'][0]['message']['content']
                return response
            
class OpenAISync(SyncLLMInterface):
    url: str = "https://api.openai.com/v1/chat/completions"
    api_key: str
    def __init__(self, api_key: str):
        self.api_key = api_key

    def getResponse(
                    self,messages: List[Dict[str,str]],
                    properties: Union[Dict,None],
                    temperature: int = 0,
                    stream: bool = False
                    ) -> Union[Union[Dict,str],Generator[str,None,None]]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data: Dict
        if properties is not None:
            # print(properties)
            data = {
                "model": "gpt-4o-mini",
                'messages': messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "QnA",
                        "schema": {
                            "type": "object",
                            "properties": properties,
                            "required": list(properties.keys()),
                        }
                    }
                },
                'temperature': temperature,
            }

            response = requests.post(
                                    self.url,
                                    headers=headers,
                                    json=data,
                                    timeout=1000000
                                    )
            response = response.json()['choices'][0]['message']['content']
            try:
                jsonOutput = json.loads(response)
            except Exception as e:
                print(response)
                raise e
            return jsonOutput
        else:
            data = {
                "model": "gpt-4o-mini",
                'messages': messages,
                'temperature': temperature,
            }
            response = requests.post(
                                    self.url,
                                    headers=headers,
                                    json=data,
                                    timeout=1000000,
                                    )
            response = response.json()['choices'][0]['message']['content']
            return response