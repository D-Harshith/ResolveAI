# ResolveAI: AI-Powered Customer Support Chatbot

## Project Overview

ResolveAI is an AI-driven customer support chatbot designed to assist customers with questions related to company policies, such as shipping, returns, refunds, and general inquiries. It uses a unified, stateless agent built with Google's Agent Development Kit (ADK) and the Gemini large language model (LLM) to provide empathetic, accurate, and efficient responses. The chatbot processes each user message independently, leveraging a built-in knowledge base, custom tools for data handling, and a local SQLite database for storing interaction histories.

For this implementation, Zappos policies were used as a reference since Zappos is considered a benchmark in customer support and return policies, emphasizing 24/7 availability, empowerment of agents, and customer happiness. The chatbot can be easily adapted by updating the policy knowledge base to work with other companies.

## Why This Project? (Problem It Solves)

Many traditional chatbots only provide support to a limited extent, often requiring human intervention in the backend for complex queries or escalations. This project aims to create a more independent system that operates without needing another person in the backend, handling routine customer inquiries autonomously.

Key problems addressed:
- **Scalability and Efficiency**: Automates responses to common questions (e.g., policy details, order issues) to reduce wait times and human workload.
- **Privacy Protection**: Automatically detects and redacts personally identifiable information (PII) like emails and phone numbers.
- **Personalized Interactions**: Maintains customer history in a database for context-aware responses without session dependencies.
- **Adaptability**: The policy knowledge base can be updated to support other companies, making it versatile beyond the Zappos reference.
- **Empathy and Compliance**: Ensures responses align with best-in-class support philosophies (inspired by Zappos) while being concise and helpful.

By building an independent AI agent, ResolveAI demonstrates how advanced LLMs and tools can elevate customer support to be more proactive and self-sufficient.

## Tools and Technologies Used

- **Google Agent Development Kit (ADK)**: Framework for agent creation, including `Agent`, `LlmAgent`, `Runner`, and `FunctionTool` for custom functionalities like ticket generation and policy retrieval.
- **Google Generative AI (Gemini)**: LLM backend (`gemini-2.5-flash-lite` model) for intent analysis, response generation, and natural language processing, with retry configurations for reliability.
- **SQLite3**: Local database (`zappos_support.db`) for storing and retrieving customer support histories.
- **python-dotenv**: Loads environment variables (e.g., `GOOGLE_API_KEY`) from a `.env` file.
- **asyncio**: Enables asynchronous agent execution.
- **logging**: Handles debug, info, and error logging.
- **re (Regular Expressions)**: For PII tokenization and policy extraction from the knowledge base.
- **uuid**: For generating unique ticket IDs.
- **Standard Python Libraries**: `os`, `time`, `sys`, `io`, `contextlib` for system operations, timing, output redirection, and context management.

The code is contained in `ResolveAI.py`. Dependencies are listed in `requirements.txt` for easy installation.

## How to Use

### Prerequisites
- Python 3.8+.
- Google API Key: Obtain from Google AI Studio or Vertex AI.
- Install dependencies from `requirements.txt`:
  ```
  pip install -r requirements.txt
  ```

### Setup
1. Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
2. Run the script:
   ```
   python ResolveAI.py
   ```
   - This initializes the SQLite database if needed.

### Usage Steps
1. **Start the Chat**:
   - Enter your name and a valid email when prompted.

2. **Interact**:
   - Type your query (e.g., "What is the return policy?").
   - The agent:
     - Analyzes intent.
     - Uses tools to generate tickets, fetch history/policies, and redact PII.
     - Provides a refined, empathetic response.
     - Saves the interaction summary.

3. **Exit**:
   - Type `quit`.

### Example Interaction
```
Welcome! Please enter your information to begin
Your Name: Jane Smith
Your Email: jane@example.com

Chat started. Type 'quit' to exit.

You: How do I return an item?
Assistant: [Empathetic response with policy details and a ticket ID]
```

## What Can Be Done Next (Future Improvements)

- **Policy Customization**: Update the `POLICY_KNOWLEDGE_BASE` constant to integrate policies from other companies, enabling multi-company support.
- **API Integrations**: Connect to real company APIs (e.g., for order tracking or inventory) to handle live data.
- **Advanced RAG**: Replace the hardcoded knowledge base with a vector database (e.g., using FAISS) for dynamic policy querying.
- **Multi-Modal Features**: Add support for image uploads (e.g., damaged items) using Gemini's vision capabilities.
- **Deployment**: Package as a web service (e.g., via Flask/Docker) for integration with websites or apps.
- **Enhanced Analytics**: Log interactions for metrics on query types and resolution success.
- **Error Resilience**: Improve handling for edge cases like invalid inputs or API failures.
- **Testing**: Add unit/integration tests to `requirements.txt`-listed dependencies.

ResolveAI is a flexible foundation for autonomous customer support—adapt and expand as needed!