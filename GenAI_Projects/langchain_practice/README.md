# LangChain Practice Repo

#### A beginner-friendly repository to start learning **LangChain** and explore the world of **Generative AI (GenAI).**

* This repo contains **beginner-friendly practice projects** as well as **advanced-level projects**.

* The example guide starts with **`try.py`**.

* In this guide, I use **Gemini** and **Composio**, but you can switch to any other LLM/toolkit by simply changing the credentials and API key.
  
  👉 The **agent logic** remains the same unless you explicitly add something new.

* Small reminder: I faced a lot of issues with **API-based tools**, so most of the projects here use **MCP server** (via [Composio](https://composio.dev)) instead of direct APIs.

* You can also use **LangChain toolkits**. They have a wide range of tools:

  * Example: `DuckDuckGoSearch` is great for prototyping and is free.
  * But note: it’s **not reliable for hackathons or production**.

* This repo is mainly for those who are **practicing GenAI for hackathons** 🚀.

---

### ℹ️ Why MCP Server?

I **highly recommend** you start using MCP servers instead of direct APIs.

* **What is an MCP server?**
  Think of MCP servers as **tools with reasoning ability**. The agent can automatically decide **which tool to use and when** — no need to hardcode tool selection with extra prompts.

* **Ways to use MCP server:**

  1. `stdio`
  2. `streamable` (recommended ✅ because it’s easier to use)

* **Providers:**

  * There are many MCP server providers.
  * I personally use **[Composio](https://composio.dev)**.
