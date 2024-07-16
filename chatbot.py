#!/usr/bin/env python
import argparse
from dotenv import load_dotenv
from google.cloud import firestore
from omegaconf import OmegaConf
from rich.markdown import Markdown

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from utils import *

console = setup()


def initialize_simple_chat(mode="local"):
    system_message = SystemMessage(
        content="""
            You are T.E.D.D.I., an AI discord robot. You were created by Teddy, who's real name is Nicole.
            You know that Teddy has a German Shepherd mix named Luna, and a long-haired cat named Meepo. 
            You have a fiance who's name is Ben (Benjamin). You are knowledge about various STEM topics, particularly
            large language modelling.
            
            Respond in a friendly way. Keep explanations simple and concise. If you do not know the answer,
            say you do not know.
            """
    )
    console.log("Initializing Chat Message History...")
    # if mode == "local":
    chat_history = ChatMessageHistory(messages=[system_message])
    # elif mode == "firebase":
    #     session_id = os.getenv("SESSION_ID")
    #     project_id = os.getenv("PROJECT_ID")
    #     collection_name = os.getenv("COLLECTION_NAME")

    #     client = firestore.Client(project=project_id)
    #     chat_history = FirestoreChatMessageHistory(
    #         session_id=session_id,
    #         collection=collection_name,
    #         client=client,
    #     )
    # console.log("Chat History Initialized.")
    console.log("Current Chat History:", chat_history.messages)

    return chat_history


def create_llm():
    if mode == "anthropic":
        llm = ChatAnthropic(model=model_name)

    elif mode == "huggingface":
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        if adapter_name:
            model.load_adapter(adapter_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        pipeline_kwargs = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            # "load_in_4bit": True,
            "max_new_tokens": 128,
            "model": model,
            "tokenizer": tokenizer,
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }
        llm = pipeline("text-generation", **pipeline_kwargs)
        # _llm = HuggingFacePipeline(pipeline=pipe)
        # llm = ChatHuggingFace(llm=_llm, verbose=True)
    return llm


def simple_chat():
    # llm = create_llm(mode="anthropic", model_name="claude-3-sonnet-20240229")
    llm = create_llm()

    # Initialize chat history
    chat_history = initialize_simple_chat()
    console.clear()
    console.print("Start chatting with the AI. Type 'exit' to quit.")

    if mode == "anthropic":

        def anthropic_simple_chat():
            while True:
                _human_input = console.input("[bold cyan]User:[/bold cyan] ")
                human_input = HumanMessage(content=_human_input)
                if human_input.content.lower() == "exit":
                    break

                chat_history.add_user_message(human_input)

                ai_response: AIMessage = llm.invoke(chat_history.messages)
                ai_context = ai_response.content
                chat_history.add_ai_message(ai_context)

                ai_context = Markdown(ai_context)
                console.print(ai_context)

        anthropic_simple_chat()
    elif mode == "huggingface":

        def huggingface_simple_chat():
            chat = [{"role": "system", "content": chat_history.messages[0].content}]
            while True:
                _human_input = console.input("[bold cyan]User:[/bold cyan] ")
                human_input = {"role": "user", "content": _human_input}
                chat.append(human_input)
                response = llm(chat, max_new_tokens=512)
                print(response[0]["generated_text"][-1]["content"])

        huggingface_simple_chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "A chatbot version of 'TEDDI'. For discord app, use 'app.py'."
    )
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
    model_name: str = model_cfg.model_name
    adapter_name: str = model_cfg.adapter_name
    mode: str = model_cfg.mode
    simple_chat()
