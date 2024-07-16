import os
import json
import pandas as pd
from dotenv import load_dotenv
from rich import pretty, traceback
from rich.console import Console


def setup(**kwargs):
    load_dotenv(".env")

    def setup_rich():
        console = Console()
        pretty.install(console=console)
        traceback.install(console=console)

        return console

    console = setup_rich()
    return console


def format_data(data_file: str, model_name: str = "nous-hermes2"):
    df = pd.read_csv(data_file)
    data_samples = []
    user_message = "You are an assistant named T.E.D.D.I. created by Teddy. Generate a response in the tone of Teddy."
    for i, row in df.iterrows():
        response: str = row["text"]
        # filter out images
        for ext in (".png", ".jpg", ".gif", ".jpeg"):
            if ext in response:
                continue

        if model_name == "nous-hermes2":
            template_response = (
                f"<s>### Instruction:\n{user_message}\n\n### Response:\n{response}</s>"
            )
        else:
            system_prompt = (
                "You are T.E.D.D.I., an AI discord robot. You were created by Teddy, who's real name is Nicole. "
                "You know that Teddy has a German Shepherd mix named Luna, and a long-haired cat named Meepo. "
                "You have a fiance who's name is Ben (Benjamin). Respond in a friendly way. "
                "Keep explanations simple and concise. If you do not know the answer, say you do not know."
            )
            template_response = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST] {response} </s>"
            data_samples.append(template_response)

    data_samples = [{"inputs": sample} for sample in data_samples]
    return data_samples


# console = setup()
# data_file = ".data/csv/discord-4nkq7191ray.csv"
# model_name = "llama2-7b-chat"
# file_name = data_file.split("/")[-1][:-4]

# formatted_data = format_data(data_file, model_name)

# import jsonlines

# json_file = f".data/json/{file_name}-{model_name}.jsonl"
# with jsonlines.open(json_file, "w") as writer:
#     writer.write_all(formatted_data)
