from typing import List, Dict, Union, Generator, AsyncGenerator
import asyncio
from .LLMInterfaces import AsyncLLMInterface, LLMInterface
from .LLMInterfaces.LLamaCPPServer import LlamaCPPServer, AsyncLlamaCPPServer

def generate_response(
    llmInterface: LLMInterface,
    messages: List[Dict[str, str]],
    properties: Union[Dict, None] = None,
    temperature: int = 0,
    stream: bool = False
) -> Union[Dict, str, Generator[str, None, None]]:
    if isinstance(llmInterface, AsyncLLMInterface):
        if stream:
            raise Exception("Stream is not supported for async interfaces as a sync generator, use async_generate_response instead")
            def stream_response() -> Generator[str, None, None]:
                async_generator_ = asyncio.run(llmInterface.getResponse(messages, properties, temperature, stream))
                print(async_generator_)
                # gen = async_generator()
                while True:
                    try:
                        # TODO fix, this ends the stream after the first token
                        yield asyncio.run(async_generator_.__anext__())
                    except StopAsyncIteration:
                        print("Stream ended")
                        break
            return stream_response()
        else:
            return asyncio.run(llmInterface.getResponse(messages, properties, temperature, stream))
    else:
        return llmInterface.getResponse(messages, properties, temperature, stream)

async def async_generate_response(
    llmInterface: AsyncLLMInterface,
    messages: List[Dict[str, str]],
    properties: Union[Dict, None] = None,
    temperature: int = 0,
    stream: bool = False
) -> Union[Union[Dict, str], AsyncGenerator]:
    if stream:
        async def stream_response() -> AsyncGenerator:
            async_generator = await llmInterface.getResponse(messages, properties, temperature, stream)
            async for token in async_generator:
                yield token
        return stream_response()
    else:
        response = await llmInterface.getResponse(messages, properties, temperature)
        return response
    

if __name__ == "__main__":
    url_ = "http://localhost:8080/v1/chat/completions"
    llamaServer = LlamaCPPServer(url_)

    messages = [
        {
            "role": "user",
            "content": "What is a freelancer?",
        }
    ]

    # # sync stream example *Perfect*
    # for token in generate_response(llamaServer, messages, None, temperature=0, stream=True):
    #     print(token)
    # print()

    # sync non-stream example *Perfect*
    response = generate_response(llamaServer, messages, None, temperature=0, stream=False)
    print(response)

    # # sync non-stream example
    # response = generate_response(llamaServer, messages, None, temperature=0, stream=False)
    # print(response)

    # # async stream example *Needs to be wrapped in a function to work*
    # # TODO make it so it is not necessary to use await async_generate_response and just use async for 
    # async def async_stream_example():
    #     async_llamaServer = AsyncLlamaCPPServer(url_)
    #     result = await async_generate_response( async_llamaServer, messages, None, temperature=0, stream=True)
    #     async for token in result:
    #             yield token
    # async def async_function():
    #     async for token in async_stream_example():
    #         print(token)
    #     print()
    # asyncio.run(async_function())

    # # async non-stream example *Perfect*
    # response = asyncio.run(async_generate_response(AsyncLlamaCPPServer(url_), messages, None, temperature=0, stream=False))
    # print(response)