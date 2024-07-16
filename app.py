import argparse
import discord
import hydra
import logging
import os
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from rich.markdown import Markdown

# discord
from discord import Message
from discord.ext import commands

# langchain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFacePipeline

from chatbot import initialize_simple_chat, create_llm
from utils import *

console = setup()
logger = logging.getLogger("discord")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
chat_history = initialize_simple_chat()


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
            # model_name = "claude-3-sonnet-20240229"
            # adapter_name = None
            llm = create_llm(
                mode=mode, model_name=model_name, adapter_name=adapter_name
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


def main():
    level = logging.INFO
    logger.setLevel(level)
    handler = RichHandler(level=level, console=console)
    formatter = logging.Formatter("%(asctime)s-%(levelname)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    client.run(os.getenv("BOT_TOKEN"), log_handler=handler, log_level=level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Runs the TEDDI bot application on Discord.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/local.yaml",
        help="Path to the configuration file.",
    )
    args: argparse.Namespace = parser.parse_args()
    config_file: str = args.config_file
    cfg = OmegaConf.load(config_file)
    model_cfg: dict = cfg.llm
    model_name = model_cfg.model_name
    adapter_name = model_cfg.adapter_name
    mode = model_cfg.mode
    main()
