# AI-as-Infrastructure Vector Store Creation

This repository contains Jupyter notebooks for creating customizable vector stores using various embedding models and indexing algorithms. The notebooks produces vector stores and associated manifest files that are compatible with the [AI as Infrastructure](https://aiinfra.anu.edu.au) [ATLAS](https://github.com/AI-as-Infrastructure/aiinfra-atlas) (Analysis and Testing of Language Models for Archival Systems) LLM RAG system. australian historic hansard debates api


## Overview

The notebooks in this repository process Hansard debates from 1901 in Australia, New Zealand, and the United Kingdom, creating vector databases with rich metadata. Each text chunk is embedded using configurable embedding models and stored in Redis with your choice of indexing algorithm (FLAT, HNSW, etc.). The repository includes examples for processing historical parliamentary texts, but can be adapted for various document types and domains.

## Features

- Supports multiple embedding models (e.g., "Livingwithmachines/bert_1890_1900", "all-mpnet-base-v2", or any Hugging Face model)
- Configurable indexing algorithms including FLAT (exact search) and HNSW (approximate search for better performance)
- Customizable document processing with rich metadata extraction (dates, URLs, page numbers)
- Supports batch processing with checkpointing for resilience
- Generates comprehensive statistics about the vector store
- Flexible configuration to adapt to different use cases and requirements

## Components

- Docker with Redis Stack
- Hugging Face Transformers for embeddings
- LangChain for document processing
- Python for orchestration

## Requirements

- Docker
- Python 3.10+
- Dependencies listed in `requirements.txt`

## Instructions

1. **Environment Setup**:
   - Create a virtual environment: `python -m venv .venv`
   - Activate it: `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
   - Install requirements: `pip install -r requirements.txt`

2. **Running the Notebooks**:
   - Open the desired notebook in VS Code or Jupyter
   - Select the `.venv` kernel
   - Execute the cells to create the vector store

3. **Vector Store Creation**:
   - The notebook will start a Redis container
   - Process documents and create embeddings
   - Store vectors with metadata in Redis
   - Generate statistics about the vector store
   - Save the database to an RDB file


## Data Sources and Licensing

The notebook is designed to process historical parliamentary debates (Hansard) from 1901. These sources are freely available, but have different licensing terms:

- **Australian Hansard**: Available at the [Parliament of Australia](https://www.aph.gov.au/parliamentary_business/hansard/hansreps_2011) and [Historic Hansard](https://www.historichansard.net/). They are licensed by the Parliament of Australia under a [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/4.0/). Specialist advice on their use to generate vector stores for the [AI as Infrastructure](https://aiinfra.anu.edu.au) was obtained.

- **Aotearoa New Zealand Hansard**: Available at the [HathiTrust Digital Library](https://babel.hathitrust.org/cgi/mb?a=listis;c=71329709) and licensed as [CC0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain). Code used to transcribe and clean the Google Books pdfs is available [here](https://github.com/AI-as-Infrastructure/aiinfra-nzhansard-preparation).

- **United Kingdom Hansard**: Available at the [History of Parliament Project](https://api.parliament.uk/historic-hansard/index.html) under [UK Parliamentary Copyright](https://en.wikipedia.org/wiki/Parliamentary_copyright#Terms).

If you choose to access and use these data sources, please respect their respective licensing terms.

## License

This code was developed with assistance from AI tools. It is made available under the [MIT License](LICENSE).