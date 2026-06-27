import json

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai.types.beta import assistant


async def main():
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

    agent1 = AssistantAgent(name ="Agent1", model_client=model_client)
    agent2 = AssistantAgent(name ="Agent2", model_client=model_client)

    await Console( agent1.run_stream(task="I am sdet 2 with knowledge of ui and api tools"))

    state = await  agent1.save_state()
    with open("memory.json", "w") as f:
        json.dump(state, f,default=str)

    with open("memory.json", "r") as f:
        savedState = json.load(f)

    await agent2.load_state(savedState)

    await Console(agent2.run_stream(task="who am i?"))
    await model_client.close()

asyncio.run(main())







