from typing import Dict, List, Union, Generator, AsyncGenerator
import requests
import json
import httpx
import sys
import os

from ._LLMInterfaces import SyncLLMInterface, AsyncLLMInterface

class LlamaCPPServer(SyncLLMInterface):
    def __init__(self, url: str):
        self.url = url

    def getResponse(
                    self,messages: List[Dict[str,str]],
                    properties: Union[Dict,None],
                    temperature: int = 0,
                    stream: bool = False
                    ) -> Union[Union[Dict,str],Generator[str,None,None]]:
        assert not stream or properties is None, "Stream is only supported for responses without properties"
        headers = {
            "Content-Type": "application/json"
        }
        data: Dict
        if properties is not None:
            # print(properties)
            data = {
                'messages': messages,
                'response_format': {
                    'type': 'json_object',
                    'schema': {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties.keys()),
                    },
                },
                'json_schema': {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties.keys()),
                    },
                'temperature': temperature,
            }

            response = requests.post(
                                    self.url,
                                    headers=headers,
                                    json=data,
                                    timeout=1000000,
                                    stream=stream
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
                'messages': messages,
                'temperature': temperature,
                'stream': stream,
            }
            response = requests.post(
                                    self.url,
                                    headers=headers,
                                    json=data,
                                    timeout=1000000,
                                    stream=stream
                                    )

            if stream:
                def stream_response() -> Generator[str, None, None]:
                    print("Streaming response:")
                    try:
                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode('utf-8')
                                if decoded_line.startswith("data: "):
                                    message = decoded_line[len("data: "):]
                                    message = message.strip()
                                    message = json.loads(message)
                                    delta = message["choices"][0]["delta"]
                                    if "content" in delta:
                                        yield delta["content"]
                                    else:
                                        break
                    except Exception as e:
                        raise e
                return stream_response()
            else:
                response = response.json()['choices'][0]['message']['content']
                return response

class AsyncLlamaCPPServer(AsyncLLMInterface):
    def __init__(self, url: str):
        self.url = url
    async def getResponse(
            self,
            messages: List[Dict[str,str]],
            properties: Union[Dict,None],
            temperature: int = 0,
            stream: bool = False
            ) -> Union[Union[Dict,str], AsyncGenerator]:
        headers = {
            "Content-Type": "application/json"
        }
        data: Dict
        if properties is not None:
            data = {
                'messages': messages,
                'response_format': {
                    'type': 'json_object',
                    'schema': {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties.keys()),
                    },
                'json_schema': {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties.keys()),
                    },
                },
                'temperature': temperature,
                "stream": stream
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
                'messages': messages,
                'temperature': temperature,
                'stream': stream
            }
            if stream:
                async def stream_response() -> AsyncGenerator[str, None]:
                    async with httpx.AsyncClient() as client:
                        async with client.stream(
                                                "POST",
                                                self.url,
                                                headers=headers,
                                                json=data,
                                                timeout=1000000
                                                ) as response:
                            async for line in response.aiter_lines():
                                print(line)
                                if line:
                                    decoded_line = line
                                    if decoded_line.startswith("data: "):
                                        message = decoded_line[len("data: "):]
                                        message = message.strip()
                                        message = json.loads(message)
                                        delta = message["choices"][0]["delta"]
                                        if "content" in delta:
                                            yield delta["content"]
                                        else:
                                            print("[async server]Stream ended")
                                            break
                return stream_response()
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                                                self.url,
                                                headers=headers,
                                                json=data,
                                                timeout=1000000)
                    response = response.json()['choices'][0]['message']
                    response = response['content']
                    return response
