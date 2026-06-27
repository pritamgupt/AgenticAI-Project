import asyncio

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from mcp.client.experimental import tasks


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

    machine = AssistantAgent(name = "machine", model_client=model_client,system_message="you behave like philosopher"
                             "when user say thanks and similar , acknowledge it and say SESSION COMPLETED to end session")
    human = UserProxyAgent(name = "human")

    humanAndMachine = RoundRobinGroupChat(participants=[machine, human], termination_condition=TextMentionTermination("SESSION COMPLETED"))

    await Console(humanAndMachine.run_stream(task = "what is life"))


asyncio.run(main())



