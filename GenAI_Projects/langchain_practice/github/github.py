from composio import Composio
from composio_langchain import LangchainProvider
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
import getpass
import os

# 1️⃣ Setup Google API key
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# 2️⃣ Initialize the LLM once
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# 3️⃣ Initialize Composio
composio = Composio(api_key="your composio api key", provider=LangchainProvider())

externalUserId = "your user id"
auth_config_id = "you are config id"

# 4️⃣ Ensure the connected account is set up (OAuth flow done earlier)
# (skip if already connected)
connected_account = composio.connected_accounts.initiate(
    user_id=externalUserId,
    auth_config_id=auth_config_id
)

# 5️⃣ Fetch GitHub-related tools
tools = composio.tools.get(
    user_id=externalUserId,
    toolkits=["github"]  # toolkit slug as lower-case / appropriate casing ("GITHUB" vs "github") may matter
)

# 6️⃣ Create the agent using new API
agent = create_react_agent(
    tools=tools,
    model=model
)

print("GitHub Agent ready! Type 'exit' to quit.")

# 7️⃣ Loop for commands
while True:
    command = input("\nEnter your GitHub command: ")
    if command.lower() in ("exit", "quit", "e", "q"):
        print("Exiting agent...")
        break

    try:
        # Key fix: Use the correct messages structure for Gemini LLMs
        result = agent.invoke({"messages": [("user", command)]})
        print("\nResponse:\n", result.get("output", result))
    except Exception as e:
        print("Error executing command:", e)
