import llama_cpp
import llama_cpp.llama_types
from ._LLMInterfaces import SyncLLMInterface, AsyncLLMInterface
import inspect
from functools import wraps
from typing import Union, List, Dict, TypedDict, Generator
import typing
import json

class LLamaCPP(SyncLLMInterface):
    llama: llama_cpp.Llama

    def __init__(self, model_path, *args, **kwargs):
        # Get the original __init__ method of llama_cpp.Llama
        llama_init = llama_cpp.Llama.__init__

        # Apply the signature of llama_init to this __init__
        signature = inspect.signature(llama_init)

        # Use wraps to retain metadata
        @wraps(llama_init)
        def wrapper_init(self, *args, **kwargs):
            # Bind arguments to the original llama_init signature
            bound_args = signature.bind(self, model_path, *args, **kwargs)
            bound_args.apply_defaults()
            
            # Initialize llama instance with bound arguments
            self.llama = llama_cpp.Llama(*bound_args.args[1:], **bound_args.kwargs)  # Skip self for llama

        # Call the wrapped init function
        wrapper_init(self, *args, **kwargs)
    
    def getResponse(self, messages: List[llama_cpp.llama_types.ChatCompletionRequestMessage], properties: Union[Dict,None], temperature: int = 0, stream: bool = False) -> Union[Dict,str,Generator,None]:
        assert not stream or properties is None, "Stream is only supported for responses without properties"
        if properties is not None:
            response = self.llama.create_chat_completion(
                messages=messages,
                response_format={
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties.keys()),
                    }
                },
                temperature=temperature
            )
            # assert response is a TypedDict
            assert isinstance(response, dict)
            response = response["choices"][0]['message']['content']
            assert isinstance(response, str)

            try:
                jsonOutput = json.loads(response)
            except Exception as e:
                print(response)
                raise e
            return jsonOutput
        else:
            response = self.llama.create_chat_completion(
                messages=messages,
                temperature=0,
                stream=stream
            )
            if stream:
                def stream_response() -> Generator[str, None, None]:
                    print("Streaming response:")
                    try:
                        for item in response:
                            assert isinstance(item, dict)
                            delta = item['choices'][0]['delta']
                            if 'content' in delta:
                                assert isinstance(delta, dict)
                                assert isinstance(delta['content'], str)
                                yield delta['content']
                            else:
                                break
                    except Exception as e:
                        raise e
                return stream_response()
            else:
                assert isinstance(response, dict)
                return response["choices"][0]["message"]["content"]
