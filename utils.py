import os
from dotenv import load_dotenv
from rich import pretty, traceback
from rich.console import Console


def setup_rich():
    console = Console()
    pretty.install(console=console)
    traceback.install(console=console)

    return console


def setup(**kwargs):
    load_dotenv(".env")
    console = setup_rich()
    return console
