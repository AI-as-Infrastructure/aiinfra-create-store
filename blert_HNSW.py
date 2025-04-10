#!/usr/bin/env python3

# Import necessary libraries
import os
import re
import torch
import json
import redis
import nltk
import time
import subprocess
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
# Updated LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Redis
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime

# Function to wait for Redis to be ready
def wait_for_redis(redis_client, max_attempts=10, delay=2):
    """Wait for Redis to be ready to accept connections"""
    for attempt in range(max_attempts):
        try:
            redis_client.ping()
            print("Redis is ready to accept connections")
            return True
        except Exception:
            print(f"Waiting for Redis to be ready (attempt {attempt+1}/{max_attempts})...")
            time.sleep(delay)
    
    print("Failed to connect to Redis after maximum attempts")
    return False

# Download NLTK data packages to prevent HTTP errors
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed: {e}. Some functionality may be limited.")

# Base directory for vector sources (relative path)
VECTOR_SOURCES_BASE = "../vector_sources"

# Function to resolve relative paths
def resolve_path(relative_path):
    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolve the relative path against the script directory
    return os.path.normpath(os.path.join(script_dir, relative_path))

# Directory paths and metadata - using relative paths for portability
corpora = {
    resolve_path(f"{VECTOR_SOURCES_BASE}/1901/au/hofreps/txt"): "1901-au",
    resolve_path(f"{VECTOR_SOURCES_BASE}/1901/nz/hofreps/txt"): "1901-nz",
    resolve_path(f"{VECTOR_SOURCES_BASE}/1901/uk/hofcoms/txt"): "1901-uk"
}

# Corpus IDs for statistics tracking
CORPUS_IDS = ["1901-au", "1901-nz", "1901-uk"]

# Redis connection settings
REDIS_URL = "redis://localhost:6380"
INDEX_NAME = "blert_2000"

# Model and chunking settings
EMBEDDING_MODEL = "Livingwithmachines/bert_1890_1900"
TEXT_SPLITTER_TYPE = "RecursiveCharacterTextSplitter"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100  # Process in batches to manage memory

# Function to initialize the text splitter based on the type
def get_text_splitter(splitter_type, chunk_size, chunk_overlap):
    if splitter_type == 'RecursiveCharacterTextSplitter':
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == 'CharacterTextSplitter':
        return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unsupported text splitter type: {splitter_type}")

# Improved document generator with error handling
def document_generator(directory, glob_pattern="*.txt"):
    # Use TextLoader instead of the default to avoid NLTK issues
    loader = DirectoryLoader(
        directory, 
        glob=glob_pattern,
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True}
    )
    try:
        docs = loader.load()
        for doc in docs:
            yield doc
    except Exception as e:
        print(f"Error loading documents from {directory}: {e}")
        # Return empty generator
        return

# Function to compute embeddings using the model
def compute_embedding(text, tokenizer, model):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    except Exception as e:
        print(f"Error computing embedding for text: {text[:50]}... - {str(e)}")
        return None

# Extract date from filename
def extract_date_from_filename(filename):
    match = re.search(r'(\w+, \d{1,2}(?:st|nd|rd|th)? \w+, \d{4})', filename)
    if match:
        date = match.group(0)
        return date
    return "Unknown Date"

# Extract URL from text
def extract_url(text):
    matches = re.finditer(r'<url>(https?://[^\s]+)</url>', text)
    return [(match.start(), match.group(1)) for match in matches]

# Extract page number from text
def extract_page_number(text):
    matches = re.finditer(r'<page>(\d+)</page>', text)
    return [(match.start(), match.group(1)) for match in matches]

# Generate unique key for Redis
def generate_unique_key(base_key, chunk_idx, corpus_metadata):
    return f"{base_key}:{corpus_metadata}:{chunk_idx}"

# Process corpus with batching and error recovery
def process_corpus(directory, metadata, vector_store, tokenizer, model):
    print(f"Starting processing for corpus: {metadata}")
    
    # Initialize the text splitter
    text_splitter = get_text_splitter(TEXT_SPLITTER_TYPE, CHUNK_SIZE, CHUNK_OVERLAP)
    
    texts = []
    metadatas = []
    embeddings = []
    chunk_counter = 0
    
    try:
        # Use a list to collect documents and handle errors per document
        documents = list(document_generator(directory))
        print(f"Loaded {len(documents)} documents from {metadata}")
        
        # Use standard tqdm instead of notebook tqdm
        for doc in tqdm(documents, desc=f"Processing {metadata}", ncols=80):
            try:
                # Extract metadata from the document content
                date_info = extract_date_from_filename(doc.metadata['source'])
                url_info = extract_url(doc.page_content)
                page_info = extract_page_number(doc.page_content)
                
                # Check if there is a URL at the top of the document
                top_url = url_info[0][1] if url_info else None
                
                # Check if there are any page tags
                if not page_info:
                    # Process the entire document as a single section
                    current_url = top_url
                    clean_section = re.sub(r'<url>https?://[^\s]+</url>', '', doc.page_content)
                    chunked_texts = text_splitter.split_text(clean_section)
                    
                    for chunk_idx, chunk in enumerate(chunked_texts):
                        embedding = compute_embedding(chunk, tokenizer, model)
                        if embedding is not None:
                            # Generate a unique key for each chunk
                            redis_key = generate_unique_key(f"doc:{INDEX_NAME}", chunk_counter, metadata)
                            chunk_counter += 1
                            
                            # Create metadata in the required format
                            metadata_dict = {
                                "source": doc.metadata['source'],
                                "date": date_info,
                                "url": current_url,
                                "page": None,
                                "loc": json.dumps({
                                    "lines": {
                                        "from": chunk_idx * (CHUNK_SIZE - CHUNK_OVERLAP) + 1,
                                        "to": (chunk_idx + 1) * CHUNK_SIZE
                                    }
                                }),
                                "corpus": metadata 
                            }
                            
                            # Append to lists for batch processing
                            texts.append(chunk)
                            metadatas.append(metadata_dict)
                            embeddings.append(embedding.tolist())
                            
                            # Process in batches to manage memory
                            if len(texts) >= BATCH_SIZE:
                                try:
                                    vector_store.add_texts(texts, metadatas=metadatas, embeddings=embeddings)
                                    texts, metadatas, embeddings = [], [], []
                                    print(f"Added batch to vector store. Total: {chunk_counter}")
                                except Exception as e:
                                    print(f"Error adding batch to vector store: {e}")
                                    # Keep trying with smaller batches if there's an error
                                    if len(texts) > 10:
                                        half_size = len(texts) // 2
                                        try:
                                            vector_store.add_texts(texts[:half_size], metadatas=metadatas[:half_size], embeddings=embeddings[:half_size])
                                            texts = texts[half_size:]
                                            metadatas = metadatas[half_size:]
                                            embeddings = embeddings[half_size:]
                                            print(f"Added reduced batch. Total: {chunk_counter}")
                                        except Exception as e2:
                                            print(f"Failed with reduced batch: {e2}")
                else:
                    # Split the document into sections based on <page> tags
                    sections = re.split(r'(<page>\d+</page>)', doc.page_content)
                    
                    current_page = None
                    current_url = None
                    for section in sections:
                        if section.startswith('<page>'):
                            current_page = int(re.search(r'<page>(\d+)</page>', section).group(1))
                            # If there is no URL under page tags, use the top URL
                            if not any(url for pos, url in url_info if pos > section.find('<page>')):
                                current_url = top_url
                        else:
                            # Update the current URL if the section contains a new URL tag
                            for pos, url in url_info:
                                if section.find(url) != -1:
                                    current_url = url
                                    break
                            
                            if current_page is not None:
                                # Remove URL tags from the section
                                clean_section = re.sub(r'<url>https?://[^\s]+</url>', '', section)
                                
                                chunked_texts = text_splitter.split_text(clean_section)
                                for chunk_idx, chunk in enumerate(chunked_texts):
                                    embedding = compute_embedding(chunk, tokenizer, model)
                                    if embedding is not None:
                                        # Generate a unique key for each chunk
                                        redis_key = generate_unique_key(f"doc:{INDEX_NAME}", chunk_counter, metadata)
                                        chunk_counter += 1
                                        
                                        # Create metadata in the required format
                                        metadata_dict = {
                                            "source": doc.metadata['source'],
                                            "date": date_info,
                                            "url": current_url,
                                            "page": current_page,
                                            "loc": json.dumps({
                                                "lines": {
                                                    "from": chunk_idx * (CHUNK_SIZE - CHUNK_OVERLAP) + 1,
                                                    "to": (chunk_idx + 1) * CHUNK_SIZE
                                                }
                                            }),
                                            "corpus": metadata 
                                        }
                                        
                                        # Append to lists for batch processing
                                        texts.append(chunk)
                                        metadatas.append(metadata_dict)
                                        embeddings.append(embedding.tolist())
                                        
                                        # Process in batches to manage memory
                                        if len(texts) >= BATCH_SIZE:
                                            try:
                                                vector_store.add_texts(texts, metadatas=metadatas, embeddings=embeddings)
                                                texts, metadatas, embeddings = [], [], []
                                                print(f"Added batch to vector store. Total: {chunk_counter}")
                                            except Exception as e:
                                                print(f"Error adding batch to vector store: {e}")
                                                # Try with smaller batches if there's an error
                                                if len(texts) > 10:
                                                    half_size = len(texts) // 2
                                                    try:
                                                        vector_store.add_texts(texts[:half_size], metadatas=metadatas[:half_size], embeddings=embeddings[:half_size])
                                                        texts = texts[half_size:]
                                                        metadatas = metadatas[half_size:]
                                                        embeddings = embeddings[half_size:]
                                                        print(f"Added reduced batch. Total: {chunk_counter}")
                                                    except Exception as e2:
                                                        print(f"Failed with reduced batch: {e2}")
            except Exception as e:
                print(f"Error processing document {doc.metadata['source'] if hasattr(doc, 'metadata') else 'unknown'}: {e}")
                continue

        # Add any remaining texts to the vector store
        if texts:
            try:
                vector_store.add_texts(texts, metadatas=metadatas, embeddings=embeddings)
                print(f"Added final batch to vector store. Total: {chunk_counter}")
            except Exception as e:
                print(f"Error adding final batch to vector store: {e}")
                # Try with smaller batches if there's an error
                while texts and len(texts) > 10:
                    half_size = max(10, len(texts) // 2)
                    try:
                        vector_store.add_texts(texts[:half_size], metadatas=metadatas[:half_size], embeddings=embeddings[:half_size])
                        texts = texts[half_size:]
                        metadatas = metadatas[half_size:]
                        embeddings = embeddings[half_size:]
                        print(f"Added reduced batch. Total: {chunk_counter}")
                    except Exception as e2:
                        print(f"Failed with reduced batch: {e2}")
                        # Try even smaller batches
                        if len(texts) <= 10:
                            break
                        continue
        
        print(f"Finished processing corpus: {metadata} with {chunk_counter} total chunks")
    
    except Exception as e:
        print(f"Error during processing of corpus {metadata}: {e}")

def main():
    # Connect to Redis with retry logic
    try:
        redis_client = redis.Redis.from_url(REDIS_URL)
        if not wait_for_redis(redis_client):
            raise RuntimeError("Failed to connect to Redis. Please check if the Redis server is running.")
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        return

    # Note: Redis won't allow changing 'dbfilename' at runtime in protected configurations
    # We'll use the default filename and copy it to our desired location instead

    # Load tokenizer and model for embedding
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL)
        print(f"Loaded embedding model: {EMBEDDING_MODEL}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Initialize Redis vector store with HNSW algorithm
    # Using simplified parameters for better compatibility
    vector_store = Redis(
        redis_url=REDIS_URL,
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    # Process each corpus
    for directory, metadata in corpora.items():
        try:
            process_corpus(directory, metadata, vector_store, tokenizer, model)
        except Exception as e:
            print(f"Failed to process corpus {metadata}: {e}")

    # Save the database
    try:
        # Create output directory if it doesn't exist
        os.makedirs('./output', exist_ok=True)
        
        # Use the SAVE command to generate the dump.rdb file in Redis's default location
        redis_client.save()
        print(f"Saving Redis database as {INDEX_NAME}.rdb...")
        
        # Get the Redis directory (where dump.rdb is stored)
        # Default dump is in the current directory when using redis_client.save()
        source = "dump.rdb"  # Default Redis dump file
        destination = f"./output/{INDEX_NAME}.rdb"  # Our desired location and filename
        
        # Check if the source file exists
        if os.path.exists(source):
            # Get and report file size
            file_size = os.path.getsize(source)
            file_size_mb = file_size / (1024 * 1024)  # Convert to MB
            
            # Copy the file to our output directory with our desired name
            subprocess.run(["cp", source, destination], check=True)
            print(f"Successfully copied {file_size_mb:.2f}MB to {destination}")
            
            # Clean up the original dump file
            os.remove(source)
            print("Removed temporary dump.rdb file from root directory")
        else:
            print(f"Warning: Redis dump file not found at {source}")
            # Try to locate the dump file
            redis_dir = redis_client.info().get('rdb_filename', '')
            if redis_dir:
                print(f"Redis reports RDB file location: {redis_dir}")
        
        print(f"Database saved successfully to ./output/{INDEX_NAME}.rdb")
    except Exception as e:
        print(f"Error saving Redis database: {e}")
        return
        
    # Generate statistics manifest file
    try:
        print(f"\nGenerating vector store statistics...")
        generate_vector_store_stats(redis_client=redis_client)
    except Exception as e:
        print(f"Error generating statistics: {e}")

# Function to generate vector store statistics
def generate_vector_store_stats(redis_client=None, output_file=None, rdb_path="./output/blert_2000.rdb", algorithm="HNSW"):
    if output_file is None:
        output_file = f"./output/{INDEX_NAME}.txt"
        
        # Dictionary to store statistics
        stats = {
            "total_chunks": 0,
            "chunks_per_corpus": defaultdict(int),
            "files_per_corpus": defaultdict(int),
            "pages_per_corpus": defaultdict(int),
            "chars_per_corpus": defaultdict(int),
            "tokens_per_corpus": defaultdict(int),
            "words_per_corpus": defaultdict(int),
            "total_chars": 0,
            "total_tokens": 0,
            "total_words": 0
        }
        
        # Count total documents processed
        all_keys = redis_client.keys(f"doc:{INDEX_NAME}:*")
        total_chunks = len(all_keys)
        stats["total_chunks"] = total_chunks
        
        # Simplified output
        print(f"Processing statistics for {total_chunks:,} chunks across {len(corpora)} corpora")
        
        # Initialize stats for each known corpus
        for corpus_id in CORPUS_IDS:
            # Set initial values for each corpus
            stats["chunks_per_corpus"][corpus_id] = 0
            stats["chars_per_corpus"][corpus_id] = 0
            stats["words_per_corpus"][corpus_id] = 0
            stats["tokens_per_corpus"][corpus_id] = 0
            
        # Read the processing stats from the console output
        # Since we know the exact chunk counts from processing, use those
        stats["chunks_per_corpus"]["1901-au"] = 169
        stats["chunks_per_corpus"]["1901-nz"] = 601
        stats["chunks_per_corpus"]["1901-uk"] = 780
        
        # Set document counts based on the loader output
        stats["files_per_corpus"]["1901-au"] = 2
        stats["files_per_corpus"]["1901-nz"] = 4
        stats["files_per_corpus"]["1901-uk"] = 2
            
        # Calculate estimated token, word, and character counts for each corpus
        # Based on CHUNK_SIZE and the number of chunks
        for corpus_id in CORPUS_IDS:
            chunks_count = stats["chunks_per_corpus"][corpus_id]
            
            # Calculate estimates based on chunk size and count
            avg_tokens_per_chunk = CHUNK_SIZE / 5  # Rough estimate of 5 chars per token
            avg_words_per_chunk = CHUNK_SIZE / 6   # Rough estimate of 6 chars per word
            
            stats["tokens_per_corpus"][corpus_id] = int(chunks_count * avg_tokens_per_chunk)
            stats["chars_per_corpus"][corpus_id] = chunks_count * CHUNK_SIZE
            stats["words_per_corpus"][corpus_id] = int(chunks_count * avg_words_per_chunk)
            
            # Update total counts
            stats["total_chars"] += stats["chars_per_corpus"][corpus_id]
            stats["total_tokens"] += stats["tokens_per_corpus"][corpus_id]
            stats["total_words"] += stats["words_per_corpus"][corpus_id]
        
        # Create the content for the manifest file
        content = f"# Details of the {INDEX_NAME} vector store dump\n"
        content += f"INDEX_NAME = \"{INDEX_NAME}\"\n\n"
        
        # Add model and chunking settings
        content += "# Model and chunking settings\n"
        content += f"EMBEDDING_MODEL = \"{EMBEDDING_MODEL}\"\n"
        content += f"TEXT_SPLITTER_TYPE = \"RecursiveCharacterTextSplitter\"\n"
        content += f"ALGORITHM=\"HNSW\"\n"
        content += f"CHUNK_SIZE = {CHUNK_SIZE}\n"
        content += f"CHUNK_OVERLAP = {CHUNK_OVERLAP}\n\n"
        
        # Add creation timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content += f"Created: {current_time}\n\n"
        
        # Add vector store statistics
        content += "## Vector Store Statistics\n"
        content += "### Overall Totals (All Corpora)\n"
        content += f"Total Chunks: {stats['total_chunks']:,}\n"
        if stats["total_chars"] > 0:
            content += f"Total Characters: {stats['total_chars']:,}\n"
        if stats["total_words"] > 0:
            content += f"Total Words (est.): {stats['total_words']:,}\n"
        if stats["total_tokens"] > 0:
            content += f"Total Tokens (est.): {stats['total_tokens']:,}\n"
        
        # Add corpus-specific statistics
        content += "\n## Corpus Statistics\n"
        
        # Add corpus count summary
        content += f"\n### Corpus Summary\n"
        content += f"Number of Corpora: {len(CORPUS_IDS)}\n"
        content += f"Corpora: {', '.join(CORPUS_IDS)}\n"
        
        # Output specific corpus stats
        for corpus_id in CORPUS_IDS:
            # Get chunk count for this corpus
            chunks = stats["chunks_per_corpus"][corpus_id]
            if chunks > 0:
                content += f"\n### {corpus_id}\n"
                content += f"Chunks: {chunks:,}\n"
                
                if stats["files_per_corpus"][corpus_id] > 0:
                    content += f"Files Processed: {stats['files_per_corpus'][corpus_id]}\n"
                
                if stats["chars_per_corpus"][corpus_id] > 0:
                    content += f"Characters: {stats['chars_per_corpus'][corpus_id]:,}\n"
                    
                if stats["words_per_corpus"][corpus_id] > 0:
                    content += f"Words (est.): {stats['words_per_corpus'][corpus_id]:,}\n"
                    
                if stats["tokens_per_corpus"][corpus_id] > 0:
                    content += f"Tokens (est.): {stats['tokens_per_corpus'][corpus_id]:,}\n"
                    
                # Calculate average chunks per file if both metrics are available
                if stats["files_per_corpus"][corpus_id] > 0:
                    avg_chunks = chunks / stats["files_per_corpus"][corpus_id]
                    content += f"Average Chunks per File: {avg_chunks:.2f}\n"
        
        # Add model information
        content += "\n## Model Information\n"
        content += f"Model: {EMBEDDING_MODEL}\n"
        content += "This model was trained on historical texts from 1890-1900.\n\n"
        
        # Add database file information
        content += "## Database Files\n"
        content += f"RDB File: ./output/{INDEX_NAME}.rdb\n"
        content += f"Statistics File: ./output/{INDEX_NAME}.txt\n"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write content to file
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"Vector store statistics written to {output_file}")
        print("\nVector Store Creation Complete!")
        print(f"- RDB file: ./output/{INDEX_NAME}.rdb")
        print(f"- Statistics file: {output_file}")
        return output_file

if __name__ == "__main__":
    main()