# Retrieval-Augmented Generation and Document Management

This repository contains code for performing Retrieval-Augmented Generation (RAG) tasks and managing documents using various utility functions. The project involves setting up the environment, configuring logging, handling document retrieval, and leveraging large language models (LLMs) for advanced data retrieval and generation.

## Features
- **Environment Configuration:** Setup for loading environment variables and configuring logging.
- **Document Retrieval:** Retrieve relevant documents based on a given query.
- **Answer Generation:** Generate answers using LLMs based on the retrieved documents.
- **Document Relevance Assessment:** Grade the relevance of documents to improve retrieval accuracy.
- **Caching Mechanisms:** Efficient caching and management of document data.
  
**Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/retrieval-augmented-generation.git
   cd retrieval-augmented-generation
   pip install -r requirements.txt
   Openai_API_key=your_openai_api_key
   ```
## Files Overview
- **graph.py:** Core functions for initializing and managing the document retrieval process.
- **Nodes_edges.py:** Functions related to handling nodes and edges within a graph structure.
- **States.py:** State management for the retrieval and generation processes.
- **Assessment_functions.py::**  Functions to assess the relevance of documents to the given query.
- **cache.py:**  Caching mechanisms for document storage and retrieval.
- **ChacheManager.py:**  Manager for handling cache operations.
- **file_reader.py:**  Functions for reading input files and managing file data.
- **Generation.py:**  Functions for setting up question re-writing and answer generation using LLMs.
