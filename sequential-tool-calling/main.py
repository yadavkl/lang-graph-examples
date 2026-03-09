from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage


# Tools
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b

def translate_to_french(text: str) -> str:
    """Return a mock French translation of the input text."""
    return f"French translation of '{text}'"

def get_weather(city: str) -> str:
    """Return a mock weather string for the given city."""
    return f"The weather in {city} is sunny."


llm = ChatOpenAI(
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_FREE_KEY"),
        temperature=0, 
        model=os.getenv("DEEPSEEK_MODEL")
    )

llm_with_tools = llm.bind_tools([multiply, translate_to_french, get_weather])

class MyMessagesState(MessagesState):
    pass

def tool_calling_llm(state: MyMessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(MyMessagesState)

builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply, translate_to_french, get_weather]))


builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition, {"tools": "tools", "default": END})
builder.add_edge("tools", "tool_calling_llm")

graph = builder.compile()

messages = graph.invoke({"messages": [HumanMessage(content="Translate the weather in Paris into French")]})
for m in messages["messages"]:
    m.pretty_print()