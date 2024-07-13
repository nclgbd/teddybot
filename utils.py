import os
from dotenv import load_dotenv
from rich import pretty, traceback
from rich.console import Console


def setup_rich():
    console = Console()
    pretty.install(console=console)
    traceback.install(console=console)

    return console


def load_envs(envs=".env/", **kwargs):
    for env in os.listdir(envs):
        load_dotenv(dotenv_path=os.path.join(envs, env))


def setup(**kwargs):
    load_envs(**kwargs)
    # project_id = os.getenv("PROJECT_ID")
    # session_id = os.getenv("SESSION_ID")
    # collection_name = os.getenv("COLLECTION_NAME")
    console = setup_rich()
    return console
