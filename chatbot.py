from dotenv import load_dotenv

from langchain.memory import ChatMessageHistory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from utils import *


def chat():
    # Create a ChatOpenAI model
    model = ChatAnthropic(model="claude-3-sonnet-20240229")

    # Initialize chat history
    console.log("Initializing Chat Message History...")
    chat_history = ChatMessageHistory()
    console.log("Chat History Initialized.")
    console.log("Current Chat History:", chat_history.messages)

    console.print("Start chatting with the AI. Type 'exit' to quit.")

    while True:
        _human_input = console.input("[bold cyan]User:[/bold cyan] ")
        human_input = HumanMessage(content=_human_input)
        if human_input.content.lower() == "exit":
            break

        chat_history.add_user_message(human_input)

        ai_response: AIMessage = model.invoke(chat_history.messages)
        chat_history.add_ai_message(ai_response.content)

        console.print(f"[bold magenta]AI: {ai_response.content}[/bold magenta]\n")


if __name__ == "__main__":
    console = setup_rich()
    load_envs()
    project_id = os.getenv("PROJECT_ID")
    session_id = os.getenv("SESSION_ID")
    collection_name = os.getenv("COLLECTION_NAME")
    console.clear()
    chat()
