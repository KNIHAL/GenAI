import getpass
import os
import asyncio
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.schema import HumanMessage, AIMessage


if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

client = MultiServerMCPClient(
    {
        "github-mcp": {
            "transport": "streamable_http",
            "url": "yor mcp server url",
        }
    }
)

async def main():
    tools = await client.get_tools()

    
    agent = create_react_agent(
        tools=tools,
        model=model
    )

    print("🚀 Github AI Assistant Ready! Type 'exit' to quit.\n")

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            response = await agent.ainvoke({"messages": chat_history + [HumanMessage(content=user_input)]})

           
            output = response.get("output", None)
            if not output and "messages" in response:
                output = response["messages"][-1].content

            print("🤖:", output)

          
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=output))

        except Exception as e:
            print("⚠️ Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
