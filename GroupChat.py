import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

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
    person1 = AssistantAgent(
        "Agent1",
        model_client=model_client,
        system_message="you are a person 1 of group chat discussion...put your thoughts on given topic"
    )

    person2 = AssistantAgent(
        "Agent2",
        model_client=model_client,
        system_message="you are person2 , counter the person1"
    )

    observer = AssistantAgent(
        "ObserverAgent",
        model_client=model_client,
        system_message="you are observer ...observe the discussion of person1 and person2 and give your conclusion and say TERMINATE"
    )
    text_termination = TextMentionTermination("TERMINATE")

    max_messages_termination = MaxMessageTermination( max_messages=8 )

    termination = text_termination |max_messages_termination

    group = SelectorGroupChat( participants=[person1, observer, person2],
                       model_client=model_client, termination_condition=termination,
                       allow_repeated_speaker=True )

    await Console(group.run_stream(task ="what your thoughts on liberalisation?" ))


asyncio.run(main())