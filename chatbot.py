#!/usr/bin/env python
from dotenv import load_dotenv
from rich.markdown import Markdown

from langchain.memory import ChatMessageHistory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFacePipeline

from utils import *

console = setup()


def initialize_simple_chat():
    system_message = SystemMessage(
        content="""
            You are T.E.D.D.I., a conversational discord bot focused on answered STEM-based questions.
            
            Respond in a friendly manner and informal way. Keep explanations simple and concise.
            """
    )
    console.log("Initializing Chat Message History...")
    chat_history = ChatMessageHistory(messages=[system_message])
    console.log("Chat History Initialized.")
    console.log("Current Chat History:", chat_history.messages)

    return chat_history


def simple_chat():
    # Create a model
    # model_name = "microsoft/Phi-3-mini-4k-instruct"
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id=model_name,
    #     task="text-generation",
    #     pipeline_kwargs={"max_new_tokens": 50},
    # )
    llm = ChatAnthropic(model="claude-3-sonnet-20240229")

    # Initialize chat history
    chat_history = initialize_simple_chat()
    console.clear()
    console.print("Start chatting with the AI. Type 'exit' to quit.")

    def _simple_chat():
        while True:
            _human_input = console.input("[bold cyan]User:[/bold cyan] ")
            human_input = HumanMessage(content=_human_input)
            if human_input.content.lower() == "exit":
                break

            chat_history.add_user_message(human_input)

            ai_response: AIMessage = llm.invoke(chat_history.messages)
            chat_history.add_ai_message(ai_response.content)

            ai_context = Markdown(ai_response.content)
            console.print(ai_context)
            # console.print(f"[bold purple]AI:[/bold purple] {ai_context}\n")

    _simple_chat()


if __name__ == "__main__":
    simple_chat()
