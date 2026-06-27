import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai.types.beta import assistant

# os.environ[
#     "OPENAI_API_KEY"] = ""
async def main():
    print("Hello world")
    # model_client = OpenAIChatCompletionClient(
    #     model="gpt-4o")

    model_client = OpenAIChatCompletionClient(
        model="llama-3.1-8b-instant",
        base_url="https://api.groq.com/openai/v1",
        api_key="",  # from console.groq.com
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        }
    )
    assistant = AssistantAgent(name="assistant", model_client=model_client)
    await Console(assistant.run_stream(task="what is 25*9 ?"))
    await model_client.close()


asyncio.run(main())

