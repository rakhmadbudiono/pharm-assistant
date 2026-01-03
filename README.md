# RAG Chatbot

Simple RAG chatbot with knowledge base upload.

## Features

- Chat with RAG (Retrieval Augmented Generation)
- Upload PDF/TXT documents to vector store
- Support for OpenAI and Gemini (free tier) models

## Installation

```bash
uv sync
```

## Configuration

Create a `.env` file in the project root:

```bash
MODEL_PROVIDER=gemini
GOOGLE_API_KEY=your-google-api-key-here
```

Or for OpenAI:

```bash
MODEL_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key-here
```

## Usage

```bash
uv run streamlit run app.py
```

1. Upload PDF or TXT files
2. Start chatting
