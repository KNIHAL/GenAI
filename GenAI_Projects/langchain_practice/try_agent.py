import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
# Ask user for a topic
user_topic = input("Enter a topic you want the AI to talk about: ")

# Initialize model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,   # make responses a bit more creative
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Messages for conversation
messages = [
    (
        "system",
        "You are a knowledgeable assistant. Give a helpful explanation to the user about their chosen topic.",
    ),
    ("human", f"Tell me about {user_topic}."),
]

# Run the model
ai_msg = llm.invoke(messages)
print(f"\nAI response about '{user_topic}':\n{ai_msg.content}")
