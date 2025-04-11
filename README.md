# langchain course agent

Proof of concept AI agent built using Langchain & OpenAI GPT model to help students with course enrollment planning.

Agent can choose between course information retriever with info generated from course catalogs, instructor information retriever with info generated from instructor reviews, and live web search API.  Agent also remembers chat history for better context.  

Usage
1. Place OpenAI & Tavily API keys in `.env`
1. `python course-planning-agent.py`
