from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="llama-3.2-3b-instruct", base_url="http://127.0.0.1:1234/v1", streaming=True)

# SystemMessage:
#   Message for priming AI behavior, usually passed in as the first of a sequenc of input messages.
# HumanMessagse:
#   Message from a human to the AI model.
messages = [
    ("system", "Solve the following math problems"),
    ("user", "What is 81 divided by 9?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")


# AIMessage:
#   Message from an AI.
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")

