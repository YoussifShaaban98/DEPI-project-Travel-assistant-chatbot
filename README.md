# DEPI-project
# Travel assistant chatbot
A Streamlit-powered AI assistant designed to answer questions about regulations and services related to activities and events in archaeological sites and museums in Egypt.
The system uses a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers from official documents.

#✨ Features

Domain-Specific Knowledge
Answers questions using a curated knowledge base of regulations and fees for:

Activities and events

Special visits

Conferences and advertising

Inspections and preparations
related to archaeological sites and museums.

RAG-Based Question Answering
Combines:

Dense vector search (Chroma + embeddings)

LLM-based generation
to ground answers strictly in the provided documents.

Arabic Language Support
Fully supports Arabic text, including long regulatory content.

Streamlit UI
Simple and interactive chat interface for asking questions and viewing responses.
📦 Requirements
Environment Variables

Create a .env file in the project root:

GOOGLE_API_KEY=your_gemini_api_key_here


Used for embeddings and LLM generation.

Python Dependencies

Install required packages:

pip install -r requirements.txt

📚 Data Format

The system uses a custom knowledge base stored in:

docs.json

✅ Data Structure

The file must be a standard JSON array (not JSONL), where each element represents one document page:

[
  {
    "document_title": "...",
    "document_type": "...",
    "page_number": 1,
    "category": "...",
    "page_content": "..."
  }
]


page_content → Main text used for retrieval and answering

Other fields → Stored as metadata to improve context and traceability

🛠️ Setup & Run

Install dependencies

pip install -r requirements.txt


Prepare the data

Place docs.json in the project root.

On first run, the app will automatically build a Chroma vector database.

Run the application

streamlit run main.py


Open in browser

http://localhost:8501

🧠 Architecture Overview

Load Documents

Reads docs.json

Converts entries into LangChain Document objects

Embedding & Storage

Documents are embedded and stored in a Chroma vector database

Retrieval

User query → vector similarity search

Most relevant document chunks are retrieved

Generation

Retrieved context + user question → LLM

Model generates a grounded, contextual answer

🗂️ Project Structure

├── main.py                 # Main Streamlit application

├── docs.json               # Regulations & fees dataset (JSON array)

├── .env                    # API keys (not tracked)

├── chroma_langchain_db/    # Vector database (auto-generated)

├── requirements.txt        # Python dependencies

└── README.md               # Project documentation

🎯 Use Cases

Asking about fees for events and activities

Understanding rules and constraints for archaeological sites

Supporting employees or visitors with quick regulatory answers

Serving as a foundation for a larger government or tourism AI assistant
