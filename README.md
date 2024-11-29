# Chatbot RAG Demo

This Demo Bot is a chatbot application designed to improve LLM-based question answering and reduce hallucinations by integrating web scraping, FAQ processing, and text-based embeddings for advanced question-answering capabilities. It utilizes Python, FastAPI, Flask, HuggingFace embeddings, and ChromaDB for semantic search and data persistence. Currently it is applied to a TA Chatbot for a highly technical course in computer science.


## Table of Contents

- [Features](#features)
- [Repo Structure](#repo-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Technical Details](#technical-details)

## Features

- **Frontend and Backend Integration**:
  - A user-friendly Flask-based frontend (`chatbot_app.py`).
  - A FastAPI backend (`app.py`) for intelligent processing and answer generation.

- **Semantic Search with ChromaDB**:
  - Employs 🤗 [HuggingFace `all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embeddings for semantic matching.
  - Uses ChromaDB for efficient vector storage and retrieval.

- **Web Scraping and Data Processing**:
  - Scrapes course-related content and FAQs with Scrapy (`scrape_website.py`).
  - Parses and organizes raw web content and FAQs into structured formats (`parse_website.py`).

- **Context-Aware Responses**:
  - Combines FAQs and content snippets to generate detailed, contextually rich answers.

- **Persistent Storage**:
  - Raw scraped data is saved for reusability.
  - Embeddings and processed data are persisted using ChromaDB.


## Repo Structure

```
.
├── chroma/                           # Directory for ChromaDB data persistence
├── raw_webcraw_data/                 # Directory for raw scraped HTML and images
├── raw_webcraw_data_faq_processed/   # Processed FAQ data from web scraping
├── templates/                        
│   └── chatbot.html                  # Chatbot frontend HTML
├── .gitignore                        # Files to ignore in Git
├── app.py                            # Main bot file
├── chatbot_app.py                    # Flask frontend
├── FAQBot.ipynb                      # Test notebook for the chatbot (deprecated)
├── parse_website.py                  # Web parsing logic for FAQ and course data
├── README_crawler.md
├── README_demobot.md
├── README.md
├── requirements.txt                  # Python dependencies
├── scrape_website.py                 # Scrapy-based web scraper
```


## Technical Details

### Semantic Search and ChromaDB

- **HuggingFace Embeddings**: The [HuggingFace `all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model is used to embed text into dense vector representations.
- **ChromaDB**: These embeddings are stored and managed using ChromaDB, enabling efficient semantic search across both course content (`chroma/db`) and FAQs (`chroma/db_faq`).

### Web Scraping and Data Preparation

- **Scrapy for Data Collection**:
  - `scrape_website.py` crawls the specified domain and collects raw HTML pages and images, stored under `raw_webcraw_data/`.
- **HTML Parsing and FAQ Extraction**:
  - `parse_website.py` processes raw HTML into structured sections, extracting FAQ question-answer pairs. Processed FAQs are saved in `raw_webcraw_data_faq_processed/`.

### Question Answering Workflow

1. **User Input**:
   - Users submit questions via the chatbot frontend.
2. **FAQ Matching**:
   - The backend searches for matching questions in the FAQ database.
3. **Content Search**:
   - Relevant course snippets are retrieved from ChromaDB using semantic search.
4. **Answer Generation**:
   - OpenAI GPT-4o-mini (Or Gemini 1.5 Flash) generates context-aware answers by combining FAQ and course snippets.

### Persistent Storage

- **Raw Data**:
  - Scraped HTML and images are stored in `raw_webcraw_data/`.
  - Processed FAQ data is saved in `raw_webcraw_data_faq_processed/`.
- **Embedding Databases**:
  - `chroma/db`: Course content embeddings.
  - `chroma/db_faq`: FAQ embeddings.


## Setup and Installation

### Prerequisites

1. Python 3.8 or higher.
2. Clone the repository:
   ```bash
   git clone https://github.com/jsz-05/Chatbot-RAG.git
   cd Chatbot-RAG
   ```
3. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   Create a `.env` file in the project root with the following content:
     ```
     OPENAI_API_KEY=<your_api_key>
     OPENAI_ORGANIZATION=<your_organization_id>
     ```

### 1. Backend API

Start the FastAPI backend:
```bash
uvicorn app:app --host 0.0.0.0 --reload
```
Access the API documentation at:
```
http://127.0.0.1:8000/docs
```

### 2. Frontend Chatbot

Run the Flask chatbot frontend:
```bash
python3 chatbot_app.py
```
Access the chatbot on:
```
http://127.0.0.1:5000/
```

