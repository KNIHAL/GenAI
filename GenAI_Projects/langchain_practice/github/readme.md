# Github Agent

### This repo has two file one uses the composio github auth and second use the composio mcp server
- You can use whatever you want. Just enter you credentials and run the file.
## ⚠️ Note: 
I initially tried using Composio Auth-based GitHub tools integration. 
While the Playground worked perfectly, the Auth configuration consistently failed 
in external/GCP environments due to connection and tool binding issues. 

Therefore, for production-ready workflows, I **highly recommend using 
Composio MCP Server instead of Auth tools**. Unless you want to spend 
hours debugging authentication, MCP Server is the more stable path.
