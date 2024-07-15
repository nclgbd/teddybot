# This example requires the 'message_content' intent.
import os
import discord
import logging
from discord import Message
from dotenv import dotenv_values
from discord.ext import commands
from rich.logging import RichHandler
from rich.markdown import Markdown

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFacePipeline

from chatbot import initialize_simple_chat, create_llm
from utils import *

console = setup()

intents = discord.Intents.default()
intents.message_content = True

# bot = commands.Bot(command_prefix="/", intents=intents)
client = discord.Client(intents=intents)


@client.event
async def on_message(message: Message):
    # this is the string text message of the Message
    content = message.content
    # this is the sender of the Message
    user = message.author
    # this is the channel of there the message is sent
    channel = message.channel

    # if the user is the client user itself, ignore the message
    if user == client.user:
        return
    if user.bot == False:
        if client.user.mentioned_in(message):
            model_name = "claude-3-sonnet-20240229"
            adapter_name = None
            llm = create_llm(
                mode="anthropic", model_name=model_name, adapter_name=adapter_name
            )
            human_input = HumanMessage(content=content)
            chat_history.add_user_message(human_input)
            async with channel.typing():
                ai_response: AIMessage = llm.invoke(chat_history.messages)
                ai_context = ai_response.content
                logger.info(ai_context)
                chat_history.add_ai_message(ai_context)

                await channel.send(ai_context)

        if message.content == "ping":
            await channel.send("pong")


if __name__ == "__main__":
    level = logging.INFO
    logger = logging.getLogger("discord")
    logger.setLevel(level)
    handler = RichHandler(level=level, console=console)
    logger.addHandler(handler)

    chat_history = initialize_simple_chat()
    discord_token = os.getenv("BOT_TOKEN")
    client.run(discord_token, log_handler=handler, log_level=level)
