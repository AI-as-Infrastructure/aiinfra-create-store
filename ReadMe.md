# Overview
This code is used to create vector stores of different kinds (primarily local Redis stores).

# Purpose
- Supports implementation of bespoke local LLM RAG apps, using Ollama + open or proprietary Chat models.
- Facilitates development of bespoke vector stores, using any Hugging Face embedding model, and production of bespoke metadata.
- React front end allows for iterative UI development as a test harness.

# Components
- Ollama with chosen model
- Redis

# Instructions
- Install Ollama and Redis if necessary.
- source .venv/bin/activate and pip install -r requirements.txt if needed.
- Run the Redis vector store pipeline, setting custom embeddings.


