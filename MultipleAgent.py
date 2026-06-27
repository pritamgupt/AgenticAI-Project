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

    agent1 = AssistantAgent(name ="Agent1", model_client=model_client, system_message="you are a curious student , ask question")
    agent2 = AssistantAgent(name ="Agent2", model_client=model_client, system_message="you are a teacher, give answer asked by student preciously in 2 lines")

    multiAgent = RoundRobinGroupChat(participants=[agent1, agent2], termination_condition=MaxMessageTermination(max_messages=3))

    await Console(multiAgent.run_stream(task= "what is llm in AI"))

    await model_client.close()

asyncio.run(main())







