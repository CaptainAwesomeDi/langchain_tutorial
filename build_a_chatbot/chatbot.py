from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory
)
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

config = dotenv_values("../.env")

os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = config["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"] = config["LANGCHAIN_API_KEY"]

model = ChatOpenAI(model="gpt-3.5-turbo")

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(model, get_session_history)

config = {"configurable": {"session_id": "abc2"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Bob")],
    config=config,
)

print(response.content)
response = with_message_history.invoke([HumanMessage("What is my name?")], config=config)
print(response.content)
