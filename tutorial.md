# Data Ingestion & Parsing for RAG Systems — LangChain v0.3

This tutorial covers everything you need to know about parsing and ingesting data for Retrieval-Augmented Generation (RAG) systems. We explore each technique with practical code examples using LangChain v0.3.

---

## Table of Contents

- [Introduction to Data Ingestion](#introduction-to-data-ingestion)
- [Understanding Document Structure in LangChain](#understanding-document-structure-in-langchain)
- [Text Files (.txt)](#text-files-txt)
  - [TextLoader — Single File](#textloader--single-file)
  - [DirectoryLoader — Multiple Files](#directoryloader--multiple-files)
- [Text Splitting Strategies](#text-splitting-strategies)
  - [CharacterTextSplitter](#1-charactertextsplitter)
  - [RecursiveCharacterTextSplitter](#2-recursivecharactertextsplitter)
  - [TokenTextSplitter](#3-tokentextsplitter)
  - [Comparison](#text-splitter-comparison)
- [PDF Documents](#pdf-documents)
  - [PyPDFLoader](#pypdfloader)
  - [PyMuPDFLoader](#pymupdfloader)
  - [PDF Loader Comparison](#pdf-loader-comparison)
- [Handling PDF Challenges](#handling-pdf-challenges)
  - [Text Cleaning](#text-cleaning)
  - [SmartPDFProcessor](#smartpdfprocessor--putting-it-all-together)
- [CSV and Excel Files — Structured Data](#csv-and-excel-files--structured-data)
  - [CSV Processing](#csv-processing)
  - [Excel Processing](#excel-processing)
- [JSON Parsing and Processing](#json-parsing-and-processing)
  - [JSONLoader with jq_schema](#method-1-jsonloader-with-jq_schema)
  - [Custom JSON Processing](#method-2-custom-json-processing--intelligent-flattening)
- [SQL Databases](#sql-databases)
  - [SQLDatabase Utility — Schema Inspection](#method-1-sqldatabase-utility--schema-inspection)
  - [Custom SQL-to-Document Conversion](#method-2-custom-sql-to-document-conversion)
- [Building a RAG System with LangChain and ChromaDB](#building-a-rag-system-with-langchain-and-chromadb)
  - [RAG Architecture Overview](#rag-architecture-overview)
  - [Loading, Splitting, Embedding](#step-1-prepare-and-load-documents)
  - [ChromaDB Vector Store](#step-4-store-embeddings-in-chromadb)
  - [Similarity Search](#step-5-similarity-search)
  - [RAG Chain — High-Level vs LCEL](#step-7-build-the-rag-chain)
  - [Adding New Documents](#adding-new-documents-to-an-existing-vector-store)
  - [Conversational RAG — Memory](#conversational-rag--memory-across-turns)
  - [Alternative LLM Providers (Groq)](#using-alternative-llm-providers-groq)

---

## Introduction to Data Ingestion

Data ingestion is the first and arguably most critical step in any RAG pipeline. Before an LLM can retrieve and reason over your data, that data must be:

1. **Loaded** from its source format (text, PDF, CSV, database, etc.)
2. **Parsed** to extract meaningful text content
3. **Split** into manageable chunks that fit within embedding model and LLM context windows
4. **Enriched** with metadata for filtering and traceability

LangChain provides a unified abstraction — the `Document` object — that standardizes how all data sources are represented after loading.

### Setup

```python
import os
from typing import List, Dict, Any
import pandas as pd

from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
```

---

## Understanding Document Structure in LangChain

Every piece of data in LangChain is represented as a `Document` object with two key fields:

| Field | Type | Description |
|-------|------|-------------|
| `page_content` | `str` | The actual text that will be embedded and searched |
| `metadata` | `dict` | Arbitrary key-value pairs for filtering, tracking, and context |

### Creating a Document

```python
doc = Document(
    page_content="This is the main text content that will be embedded and searched.",
    metadata={
        "source": "example.txt",
        "page": 1,
        "author": "Krish Naik",
        "date_created": "2024-01-01",
        "custom_field": "any_value"
    }
)

print(f"Content: {doc.page_content}")
print(f"Metadata: {doc.metadata}")
print(type(doc))  # <class 'langchain_core.documents.base.Document'>
```

### Why Metadata Matters

Metadata is not just bookkeeping — it plays an active role in RAG systems:

- **Filtering search results**: Narrow retrieval to specific sources, date ranges, or categories (e.g., "only search documents from 2024").
- **Tracking document sources**: Know exactly where a retrieved chunk came from so you can cite it or verify it.
- **Providing context in responses**: Pass metadata to the LLM alongside retrieved text so it can say things like "According to page 3 of the financial report..."
- **Debugging and auditing**: Trace back from a generated answer to the exact chunk and source file.

> **Tip**: Always include at minimum a `source` field in metadata. Other useful fields include `page`, `author`, `date`, and any domain-specific tags.

---

## Text Files (.txt)

Text files are the simplest data source. LangChain provides two loaders for them.

### Preparing Sample Data

```python
import os
os.makedirs("data/text_files", exist_ok=True)

sample_texts = {
    "data/text_files/python_intro.txt": """Python Programming Introduction

Python is a high-level, interpreted programming language known for its simplicity and readability.
Created by Guido van Rossum and first released in 1991, Python has become one of the most popular
programming languages in the world.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support

Python is widely used in web development, data science, artificial intelligence, and automation.""",

    "data/text_files/machine_learning.txt": """Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn and improve
from experience without being explicitly programmed. It focuses on developing computer programs
that can access data and use it to learn for themselves.

Types of Machine Learning:
1. Supervised Learning: Learning with labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through rewards and penalties

Applications include image recognition, speech processing, and recommendation systems"""
}

for filepath, content in sample_texts.items():
    with open(filepath, 'w', encoding="utf-8") as f:
        f.write(content)
```

### TextLoader — Single File

`TextLoader` reads a single text file and returns it as one `Document`.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/text_files/python_intro.txt", encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} document")         # 1
print(f"Content preview: {documents[0].page_content[:100]}...")
print(f"Metadata: {documents[0].metadata}")
# {'source': 'data/text_files/python_intro.txt'}
```

**What happens under the hood**: `TextLoader` opens the file, reads the entire contents into a single string, and wraps it in a `Document` with `source` metadata pointing to the file path. The entire file becomes one document — no splitting happens at this stage.

### DirectoryLoader — Multiple Files

`DirectoryLoader` scans a directory and loads all files matching a glob pattern.

```python
from langchain_community.document_loaders import DirectoryLoader

dir_loader = DirectoryLoader(
    "data/text_files",
    glob="**/*.txt",                    # Pattern to match files
    loader_cls=TextLoader,              # Loader class to use per file
    loader_kwargs={'encoding': 'utf-8'},
    show_progress=True                  # Shows a tqdm progress bar
)

documents = dir_loader.load()

print(f"Loaded {len(documents)} documents")  # 2
for i, doc in enumerate(documents):
    print(f"  Document {i+1}: {doc.metadata['source']} — {len(doc.page_content)} chars")
```

**Key parameters**:

| Parameter | Description |
|-----------|-------------|
| `glob` | File pattern — `"**/*.txt"` matches `.txt` files recursively |
| `loader_cls` | Which loader to use for each matched file |
| `loader_kwargs` | Arguments passed to each loader instance |
| `show_progress` | Display a progress bar during loading |

**Advantages**:
- Loads multiple files at once
- Supports glob patterns for flexible file matching
- Progress tracking with `show_progress=True`
- Recursive directory scanning with `**/`

**Disadvantages**:
- All files must be the same type (same `loader_cls`)
- Limited per-file error handling
- Can be memory intensive for very large directories

---

## Text Splitting Strategies

Once you've loaded documents, they're usually too large to embed or fit into an LLM's context window. **Text splitters** break documents into smaller chunks.

### Why Splitting Matters

- **Embedding models** have token limits (e.g., 512 tokens for many models).
- **Smaller, focused chunks** produce better semantic search results than large documents.
- **Chunk overlap** ensures that context isn't lost at chunk boundaries — a sentence cut in half at a boundary is preserved in the overlapping region of the next chunk.

```python
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)

text = documents[0].page_content  # The machine_learning.txt content
```

### 1. CharacterTextSplitter

Splits text on a **single separator** character. Simple and predictable.

```python
char_splitter = CharacterTextSplitter(
    separator="\n",       # Split on newlines
    chunk_size=200,       # Max chunk size in characters
    chunk_overlap=20,     # Overlap between consecutive chunks
    length_function=len   # How to measure chunk size
)

char_chunks = char_splitter.split_text(text)
print(f"Created {len(char_chunks)} chunks")  # 4 chunks
```

**How it works**:
1. Split the text on the `separator` (here, `\n`).
2. Merge adjacent pieces until `chunk_size` is reached.
3. Start the next chunk with `chunk_overlap` characters from the end of the previous chunk.

**Example output** (with `separator="\n"`):

```
Chunk 1: "Machine Learning Basics\nMachine learning is a subset of artificial intelligence..."
-----
Chunk 2: "from experience without being explicitly programmed. It focuses on developing..."
-----
Chunk 3: "1. Supervised Learning: Learning with labeled data\n2. Unsupervised Learning..."
```

With `separator=" "` instead, the splitter breaks on spaces, which can produce different chunk boundaries:

```python
char_splitter_space = CharacterTextSplitter(
    separator=" ",
    chunk_size=200,
    chunk_overlap=20,
    length_function=len
)
space_chunks = char_splitter_space.split_text(text)
print(f"Created {len(space_chunks)} chunks")  # 3 chunks
```

> **Key insight**: The choice of separator significantly affects chunk boundaries. `"\n"` respects line structure; `" "` gives finer control but may break logical sections.

### 2. RecursiveCharacterTextSplitter

Tries **multiple separators in order of priority**, falling back to the next one if chunks are still too large. This is the **recommended default splitter** for most use cases.

```python
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],  # Try these in order
    chunk_size=200,
    chunk_overlap=20,
    length_function=len
)

recursive_chunks = recursive_splitter.split_text(text)
print(f"Created {len(recursive_chunks)} chunks")
```

**How it works**:
1. First try splitting on `"\n\n"` (paragraph breaks).
2. If any resulting piece is still > `chunk_size`, split those pieces on `"\n"` (line breaks).
3. Still too big? Split on `" "` (spaces).
4. Last resort: split on `""` (individual characters).

This approach **respects the natural hierarchy of text structure**: paragraphs > lines > words > characters.

**Example with a simple text**:

```python
simple_text = "This is sentence one and it is quite long. This is sentence two and it is also quite long. This is sentence three which is even longer than the others. This is sentence four. This is sentence five. This is sentence six."

splitter = RecursiveCharacterTextSplitter(
    separators=[" "],
    chunk_size=80,
    chunk_overlap=20,
    length_function=len
)

chunks = splitter.split_text(simple_text)
# 4 chunks, with 20-char overlaps between consecutive chunks:
# Chunk 1: 'This is sentence one and it is quite long. This is sentence two and it is also'
# Chunk 2: 'two and it is also quite long. This is sentence three which is even longer than'
#                               ^^^^ overlap from chunk 1 ^^^^
```

### 3. TokenTextSplitter

Splits based on **token count** rather than character count. Useful when working with models that have strict token limits.

```python
token_splitter = TokenTextSplitter(
    chunk_size=50,      # Size in tokens (not characters!)
    chunk_overlap=10
)

token_chunks = token_splitter.split_text(text)
print(f"Created {len(token_chunks)} chunks")  # 3 chunks
```

**How it works**: Uses a tokenizer (by default, OpenAI's `tiktoken`) to count tokens. This ensures chunks align with how the model actually processes text, avoiding silent truncation.

> **When to use**: When you need precise control over token counts — for example, if your embedding model has a hard 512-token limit, set `chunk_size=500` with some buffer.

### Text Splitter Comparison

| Splitter | Splits On | Best For | Drawback |
|----------|-----------|----------|----------|
| `CharacterTextSplitter` | Single separator | Structured text with clear delimiters | May break mid-sentence |
| `RecursiveCharacterTextSplitter` | Multiple separators (hierarchical) | **General purpose — default choice** | Slightly more complex |
| `TokenTextSplitter` | Token boundaries | Token-limited models, precise sizing | Slower (requires tokenizer) |

**Rule of thumb**: Start with `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200`. Adjust based on your embedding model's limits and retrieval quality.

---

## PDF Documents

PDFs are one of the most common document formats but also one of the hardest to parse — text can be stored as vectors, images, or encoded in complex internal structures.

LangChain provides multiple PDF loaders, each with different trade-offs.

### PyPDFLoader

The simplest PDF loader. Uses the `pypdf` library under the hood.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/pdf/attention.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages")  # 15 pages
print(f"Page 1 preview: {docs[0].page_content[:100]}...")
print(f"Metadata: {docs[0].metadata}")
```

**Metadata returned** includes:
- `source` — file path
- `page` — zero-indexed page number
- `page_label` — human-readable page label
- `total_pages` — total page count
- PDF-level metadata: `producer`, `creator`, `creationdate`, `author`, `title`, etc.

Each page becomes a separate `Document`, so a 15-page PDF produces 15 documents.

### PyMuPDFLoader

Uses the `PyMuPDF` (fitz) library. Faster and often produces better text extraction, especially for complex layouts.

```python
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("data/pdf/attention.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages")  # 15 pages
print(f"Metadata keys: {list(docs[0].metadata.keys())}")
```

PyMuPDFLoader returns richer metadata than PyPDFLoader, including `format` (e.g., `"PDF 1.5"`), `file_path`, and additional date fields.

### PDF Loader Comparison

| Loader | Library | Speed | Text Quality | Metadata | Best For |
|--------|---------|-------|-------------|----------|----------|
| `PyPDFLoader` | `pypdf` | Moderate | Good | Standard | Standard text-based PDFs |
| `PyMuPDFLoader` | `PyMuPDF` | **Fast** | **Better** | Detailed | Speed-critical pipelines, complex layouts |

> **Installation**: `pip install pypdf` for PyPDFLoader, `pip install pymupdf` for PyMuPDFLoader.

---

## Handling PDF Challenges

PDFs are notoriously difficult to parse because they:

- Store text in complex internal representations (not just simple strings)
- Can have formatting artifacts and broken whitespace
- May contain scanned images (requiring OCR)
- Often have ligature encoding issues (e.g., `ﬁ` instead of `fi`)

### Text Cleaning

A common post-processing step after loading PDFs:

```python
def clean_text(text):
    # Remove excessive whitespace (multiple spaces, tabs, newlines)
    text = " ".join(text.split())

    # Fix common ligature issues from PDF extraction
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")

    return text
```

**Example**:

```python
raw = """Company Financial Report


    The ﬁnancial performance for ﬁscal year 2024
    shows signiﬁcant growth in proﬁtability.



    Revenue increased by 25%.

The company's efﬁciency improved due to workﬂow
optimization.


Page 1 of 10
"""

cleaned = clean_text(raw)
# "Company Financial Report The financial performance for fiscal year 2024 shows
#  significant growth in profitability. Revenue increased by 25%. The company's
#  efficiency improved due to workflow optimization. Page 1 of 10"
```

### SmartPDFProcessor — Putting It All Together

A reusable class that combines loading, cleaning, chunking, and metadata enrichment:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class SmartPDFProcessor:
    """Advanced PDF processing with cleaning, chunking, and metadata enrichment."""

    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Load, clean, chunk, and enrich a PDF."""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        processed_chunks = []
        for page_num, page in enumerate(pages):
            cleaned_text = self._clean_text(page.page_content)

            # Skip nearly empty pages (e.g., blank separator pages)
            if len(cleaned_text.strip()) < 50:
                continue

            chunks = self.text_splitter.create_documents(
                texts=[cleaned_text],
                metadatas=[{
                    **page.metadata,
                    "page": page_num + 1,
                    "total_pages": len(pages),
                    "chunk_method": "smart_pdf_processor",
                    "char_count": len(cleaned_text)
                }]
            )
            processed_chunks.extend(chunks)

        return processed_chunks

    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text."""
        text = " ".join(text.split())
        text = text.replace("ﬁ", "fi")
        text = text.replace("ﬂ", "fl")
        return text
```

**Usage**:

```python
processor = SmartPDFProcessor(chunk_size=1000, chunk_overlap=100)
chunks = processor.process_pdf("data/pdf/attention.pdf")

print(f"Processed into {len(chunks)} chunks")  # 49 chunks

# Each chunk has enriched metadata:
for key, value in chunks[0].metadata.items():
    print(f"  {key}: {value}")
# source: data/pdf/attention.pdf
# page: 1
# total_pages: 15
# chunk_method: smart_pdf_processor
# char_count: 2857
```

**Why this is better than raw loading**:
1. **Cleaned text** — no ligature issues or whitespace noise.
2. **Smart chunking** — uses `RecursiveCharacterTextSplitter` for semantically meaningful chunks.
3. **Empty page filtering** — skips blank/separator pages that would add noise.
4. **Enriched metadata** — every chunk knows its page number, total pages, character count, and processing method, making downstream filtering and debugging straightforward.

---

## CSV and Excel Files — Structured Data

Structured data in CSV and Excel files requires a different approach from free-text documents. Each row represents a distinct record, and the column headers provide schema context that raw text loaders would discard.

The key challenge: how do you turn tabular data into `Document` objects that an LLM can reason over effectively?

### Preparing Sample Data

```python
import pandas as pd
import os

os.makedirs("data/structured_files", exist_ok=True)

data = {
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam'],
    'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics'],
    'Price': [999.99, 29.99, 79.99, 299.99, 89.99],
    'Stock': [50, 200, 150, 75, 100],
    'Description': [
        'High-performance laptop with 16GB RAM and 512GB SSD',
        'Wireless optical mouse with ergonomic design',
        'Mechanical keyboard with RGB backlighting',
        '27-inch 4K monitor with HDR support',
        '1080p webcam with noise cancellation'
    ]
}

df = pd.DataFrame(data)
df.to_csv('data/structured_files/products.csv', index=False)
```

For Excel, you can create multi-sheet workbooks:

```python
with pd.ExcelWriter('data/structured_files/inventory.xlsx') as writer:
    df.to_excel(writer, sheet_name='Products', index=False)

    summary_data = {
        'Category': ['Electronics', 'Accessories'],
        'Total_Items': [3, 2],
        'Total_Value': [1389.97, 109.98]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
```

### CSV Processing

#### Method 1: CSVLoader — Row-Based Documents

`CSVLoader` creates **one Document per row**. Each row's columns are formatted as `key: value` pairs in the `page_content`.

```python
from langchain_community.document_loaders import CSVLoader

csv_loader = CSVLoader(
    file_path='data/structured_files/products.csv',
    encoding='utf-8',
    csv_args={
        'delimiter': ',',
        'quotechar': '"',
    }
)

csv_docs = csv_loader.load()
print(f"Loaded {len(csv_docs)} documents (one per row)")  # 5
```

**What a single document looks like**:

```
Content:
  Product: Laptop
  Category: Electronics
  Price: 999.99
  Stock: 50
  Description: High-performance laptop with 16GB RAM and 512GB SSD

Metadata: {'source': 'data/structured_files/products.csv', 'row': 0}
```

Each row becomes a self-contained document with the column names as labels. The metadata includes the `source` file path and the `row` index (zero-based).

**Key parameters for CSVLoader**:

| Parameter | Description |
|-----------|-------------|
| `file_path` | Path to the CSV file |
| `encoding` | Character encoding (default `utf-8`) |
| `csv_args` | Dict passed to Python's `csv.DictReader` — control `delimiter`, `quotechar`, etc. |
| `source_column` | Optionally use a column's value as the `source` in metadata instead of the file path |

#### Method 2: Custom CSV Processing — Intelligent Documents

For better RAG performance, you often want more control over how rows become documents. A custom approach lets you:

- Format `page_content` in a more natural, readable way
- Promote important columns to metadata for filtering
- Add computed fields or type annotations

```python
from typing import List
from langchain_core.documents import Document

def process_csv_intelligently(filepath: str) -> List[Document]:
    """Process CSV with structured content and rich metadata."""
    df = pd.read_csv(filepath)
    documents = []

    for idx, row in df.iterrows():
        content = f"""Product Information:
        Name: {row['Product']}
        Category: {row['Category']}
        Price: ${row['Price']}
        Stock: {row['Stock']} units
        Description: {row['Description']}"""

        doc = Document(
            page_content=content,
            metadata={
                'source': filepath,
                'row_index': idx,
                'product_name': row['Product'],
                'category': row['Category'],
                'price': row['Price'],
                'data_type': 'product_info'
            }
        )
        documents.append(doc)

    return documents

docs = process_csv_intelligently('data/structured_files/products.csv')
```

**Why this is better for RAG**:

- The `page_content` reads like natural language, which embedding models handle better than raw CSV rows.
- Key fields like `category` and `price` are in metadata, enabling filtered retrieval (e.g., "find all Electronics under $500").
- The `data_type` field lets you distinguish product records from other document types in a mixed vector store.

#### CSV Processing Strategy Comparison

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| **CSVLoader** (row-based) | Simple, zero config | Loses table context, generic formatting | Quick prototyping, record lookups |
| **Custom processing** | Rich metadata, natural language content, filterable | More code to write | Production RAG, Q&A over structured data |

### Excel Processing

Excel files add complexity over CSV: they can have **multiple sheets**, formatting, formulas, and merged cells.

#### Method 1: Pandas-Based Processing — Full Control

Process each sheet as a separate document, preserving sheet-level context:

```python
def process_excel_with_pandas(filepath: str) -> List[Document]:
    """Process Excel with sheet awareness."""
    documents = []
    excel_file = pd.ExcelFile(filepath)

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet_name)

        sheet_content = f"Sheet: {sheet_name}\n"
        sheet_content += f"Columns: {', '.join(df.columns)}\n"
        sheet_content += f"Rows: {len(df)}\n\n"
        sheet_content += df.to_string(index=False)

        doc = Document(
            page_content=sheet_content,
            metadata={
                'source': filepath,
                'sheet_name': sheet_name,
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'data_type': 'excel_sheet'
            }
        )
        documents.append(doc)

    return documents

excel_docs = process_excel_with_pandas('data/structured_files/inventory.xlsx')
print(f"Processed {len(excel_docs)} sheets")  # 2
```

**Example output** — the "Products" sheet becomes:

```
Sheet: Products
Columns: Product, Category, Price, Stock, Description
Rows: 5

 Product    Category  Price  Stock                                         Description
  Laptop Electronics 999.99     50 High-performance laptop with 16GB RAM and 512GB SSD
   Mouse Accessories  29.99    200        Wireless optical mouse with ergonomic design
   ...
```

And the "Summary" sheet becomes a separate document with its own metadata (`sheet_name: 'Summary'`, `num_rows: 2`, etc.).

#### Method 2: UnstructuredExcelLoader

For complex Excel files with formatting, merged cells, or embedded objects, `UnstructuredExcelLoader` from the `unstructured` library can handle features that pandas ignores.

```python
from langchain_community.document_loaders import UnstructuredExcelLoader

excel_loader = UnstructuredExcelLoader(
    'data/structured_files/inventory.xlsx',
    mode="elements"    # "elements" = one doc per table/element; "single" = one doc for everything
)
unstructured_docs = excel_loader.load()
```

In `"elements"` mode, each sheet's table becomes a separate `Document`. The metadata is notably richer:

| Metadata Field | Example Value | Description |
|----------------|---------------|-------------|
| `page_name` | `"Products"` | Sheet name |
| `page_number` | `1` | Sheet index |
| `text_as_html` | `<table>...</table>` | HTML representation of the table |
| `category` | `"Table"` | Element type detected |
| `filetype` | `application/vnd.openxmlformats-...` | MIME type |
| `languages` | `['eng']` | Detected languages |

The `text_as_html` field is particularly useful — if you need to render the table or parse it further downstream, you have the full HTML table structure preserved.

> **Installation**: `pip install unstructured openpyxl` for `UnstructuredExcelLoader`.

#### Excel Loader Comparison

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Pandas-based** | Full control, sheet-aware, no extra deps | Manual coding, misses formatting | Custom pipelines, data analysis context |
| **UnstructuredExcelLoader** | Handles complex features, HTML output, rich metadata | Requires `unstructured` library | Complex workbooks, preserving table structure |

---

## JSON Parsing and Processing

JSON is the lingua franca of APIs, config files, and NoSQL databases. Unlike flat CSV rows, JSON can be **deeply nested** — objects within arrays within objects — which makes it both powerful and tricky to ingest into a RAG pipeline.

The core challenge: how do you flatten a tree-shaped data structure into flat `Document` objects without losing the relationships between nested fields?

### Preparing Sample Data

**Nested JSON** — a company with employees, skills, projects, and departments:

```python
import json
import os

os.makedirs("data/json_files", exist_ok=True)

json_data = {
    "company": "TechCorp",
    "employees": [
        {
            "id": 1,
            "name": "John Doe",
            "role": "Software Engineer",
            "skills": ["Python", "JavaScript", "React"],
            "projects": [
                {"name": "RAG System", "status": "In Progress"},
                {"name": "Data Pipeline", "status": "Completed"}
            ]
        },
        {
            "id": 2,
            "name": "Jane Smith",
            "role": "Data Scientist",
            "skills": ["Python", "Machine Learning", "SQL"],
            "projects": [
                {"name": "ML Model", "status": "In Progress"},
                {"name": "Analytics Dashboard", "status": "Planning"}
            ]
        }
    ],
    "departments": {
        "engineering": {
            "head": "Mike Johnson",
            "budget": 1000000,
            "team_size": 25
        },
        "data_science": {
            "head": "Sarah Williams",
            "budget": 750000,
            "team_size": 15
        }
    }
}

with open('data/json_files/company_data.json', 'w') as f:
    json.dump(json_data, f, indent=2)
```

**JSON Lines (`.jsonl`)** — one JSON object per line, common for event logs and streaming data:

```python
jsonl_data = [
    {"timestamp": "2024-01-01", "event": "user_login", "user_id": 123},
    {"timestamp": "2024-01-01", "event": "page_view", "user_id": 123, "page": "/home"},
    {"timestamp": "2024-01-01", "event": "purchase", "user_id": 123, "amount": 99.99}
]

with open('data/json_files/events.jsonl', 'w') as f:
    for item in jsonl_data:
        f.write(json.dumps(item) + '\n')
```

> **JSON vs JSONL**: Standard JSON wraps everything in one object/array. JSONL stores one independent JSON object per line — it's streamable, appendable, and each line can be processed independently. Many logging systems and data pipelines use JSONL.

### JSON Processing Strategies

#### Method 1: JSONLoader with `jq_schema`

LangChain's `JSONLoader` uses **jq syntax** to extract specific parts of a JSON structure. `jq` is a lightweight query language for JSON (similar to XPath for XML).

```python
from langchain_community.document_loaders import JSONLoader

employee_loader = JSONLoader(
    file_path='data/json_files/company_data.json',
    jq_schema='.employees[]',   # jq query: iterate over each employee
    text_content=False           # False = keep full JSON objects as content
)

employee_docs = employee_loader.load()
print(f"Loaded {len(employee_docs)} employee documents")  # 2
```

**What each document looks like**:

```
Content: {"id": 1, "name": "John Doe", "role": "Software Engineer",
          "skills": ["Python", "JavaScript", "React"],
          "projects": [{"name": "RAG System", "status": "In Progress"}, ...]}

Metadata: {'source': '.../company_data.json', 'seq_num': 1}
```

**Key parameters**:

| Parameter | Description |
|-----------|-------------|
| `file_path` | Path to the JSON file |
| `jq_schema` | A jq expression that selects which parts of the JSON become documents |
| `text_content` | `True` = extract as plain text string; `False` = serialize matched objects as JSON strings |

**Common `jq_schema` patterns**:

| Pattern | What it selects |
|---------|----------------|
| `.` | The entire document as one `Document` |
| `.employees[]` | Each element in the `employees` array |
| `.employees[].name` | Just the `name` field from each employee |
| `.departments` | The entire `departments` object |
| `.employees[] \| {name, role}` | Only `name` and `role` from each employee |

> **Installation note**: `JSONLoader` requires the `jq` Python package: `pip install jq`.

#### Method 2: Custom JSON Processing — Intelligent Flattening

For production RAG systems, you typically want human-readable `page_content` and rich metadata — not raw JSON strings. A custom processor gives you full control over how nested structures are flattened.

```python
from typing import List
from langchain_core.documents import Document

def process_json_intelligently(filepath: str) -> List[Document]:
    """Process JSON with intelligent flattening and context preservation."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    documents = []

    for emp in data.get('employees', []):
        content = f"""Employee Profile:
        Name: {emp['name']}
        Role: {emp['role']}
        Skills: {', '.join(emp['skills'])}

        Projects:"""
        for proj in emp.get('projects', []):
            content += f"\n- {proj['name']} (Status: {proj['status']})"

        doc = Document(
            page_content=content,
            metadata={
                'source': filepath,
                'data_type': 'employee_profile',
                'employee_id': emp['id'],
                'employee_name': emp['name'],
                'role': emp['role']
            }
        )
        documents.append(doc)

    return documents

docs = process_json_intelligently("data/json_files/company_data.json")
```

**Example output**:

```
Content:
  Employee Profile:
        Name: John Doe
        Role: Software Engineer
        Skills: Python, JavaScript, React

        Projects:
  - RAG System (Status: In Progress)
  - Data Pipeline (Status: Completed)

Metadata: {'source': 'data/json_files/company_data.json',
            'data_type': 'employee_profile',
            'employee_id': 1,
            'employee_name': 'John Doe',
            'role': 'Software Engineer'}
```

**Why this approach is better for RAG**:

1. **Natural language content** — embedding models produce better vectors for readable text than for raw JSON syntax with braces and quotes.
2. **Nested data is flattened meaningfully** — skills become a comma-separated list, projects are listed with their status, preserving the relationship without the nesting complexity.
3. **Filterable metadata** — `employee_id`, `role`, and `data_type` let you do targeted retrieval (e.g., "find all Data Scientists" by filtering on `role`).
4. **Consistent structure** — every employee document has the same format, making retrieval results predictable.

#### JSON Processing Strategy Comparison

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| **JSONLoader + jq** | Concise, powerful queries, handles any structure | Raw JSON in content, requires learning jq syntax | Quick extraction, prototyping, simple schemas |
| **Custom processing** | Human-readable content, rich metadata, full control | More code, schema-specific | Production RAG, Q&A over JSON data |

> **Tip for deeply nested JSON**: Consider creating multiple document types from the same file. For example, from the company data above you could create employee profile documents, department summary documents, and project status documents — each optimized for different kinds of questions.

---

## SQL Databases

Databases are a critical data source for enterprise RAG systems. Unlike files, databases offer:

- **Structured, normalized data** spread across multiple related tables
- **Real-time access** — always up-to-date, no stale file copies
- **SQL queries** — you can pre-join, filter, and aggregate before creating documents

The challenge: relational data is spread across tables connected by foreign keys. A single "fact" (like "Jane Smith leads the ML Platform project") might require joining two or more tables to reconstruct.

### Setting Up a Sample SQLite Database

```python
import sqlite3
import os

os.makedirs("data/databases", exist_ok=True)

conn = sqlite3.connect('data/databases/company.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''CREATE TABLE IF NOT EXISTS employees
                 (id INTEGER PRIMARY KEY, name TEXT, role TEXT,
                  department TEXT, salary REAL)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS projects
                 (id INTEGER PRIMARY KEY, name TEXT, status TEXT,
                  budget REAL, lead_id INTEGER)''')

# Insert sample data
employees = [
    (1, 'John Doe', 'Senior Developer', 'Engineering', 95000),
    (2, 'Jane Smith', 'Data Scientist', 'Analytics', 105000),
    (3, 'Mike Johnson', 'Product Manager', 'Product', 110000),
    (4, 'Sarah Williams', 'DevOps Engineer', 'Engineering', 98000)
]

projects = [
    (1, 'RAG Implementation', 'Active', 150000, 1),
    (2, 'Data Pipeline', 'Completed', 80000, 2),
    (3, 'Customer Portal', 'Planning', 200000, 3),
    (4, 'ML Platform', 'Active', 250000, 2)
]

cursor.executemany('INSERT OR REPLACE INTO employees VALUES (?,?,?,?,?)', employees)
cursor.executemany('INSERT OR REPLACE INTO projects VALUES (?,?,?,?,?)', projects)

conn.commit()
conn.close()
```

### Database Content Extraction

#### Method 1: SQLDatabase Utility — Schema Inspection

LangChain's `SQLDatabase` utility wraps a database connection and provides convenient methods for inspecting schema and running queries. It's the foundation for LangChain's SQL agents and chains.

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///data/databases/company.db")

print(f"Tables: {db.get_usable_table_names()}")
# ['employees', 'projects']

print(db.get_table_info())
```

**Output of `get_table_info()`** — this is particularly valuable because it shows both the DDL schema and sample rows:

```sql
CREATE TABLE employees (
    id INTEGER,
    name TEXT,
    role TEXT,
    department TEXT,
    salary REAL,
    PRIMARY KEY (id)
)

/*
3 rows from employees table:
id   name           role              department    salary
1    John Doe       Senior Developer  Engineering   95000.0
2    Jane Smith     Data Scientist    Analytics     105000.0
3    Mike Johnson   Product Manager   Product       110000.0
*/
```

> **Why this matters for RAG**: The DDL + sample rows together give an LLM enough context to understand the schema and write SQL queries against it. This is the basis of text-to-SQL systems.

The connection string format is `"sqlite:///path/to/db"` for SQLite. For other databases, use SQLAlchemy connection strings:
- PostgreSQL: `"postgresql://user:password@host:port/dbname"`
- MySQL: `"mysql://user:password@host:port/dbname"`

#### Method 2: Custom SQL-to-Document Conversion

For RAG, you typically want to extract database content into documents that capture both individual table data and cross-table relationships.

```python
import sqlite3
from typing import List
from langchain_core.documents import Document

def sql_to_documents(db_path: str) -> List[Document]:
    """Convert SQL database to documents with context."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    documents = []

    # Strategy 1: One document per table (schema + sample data)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]

        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        table_content = f"Table: {table_name}\n"
        table_content += f"Columns: {', '.join(column_names)}\n"
        table_content += f"Total Records: {len(rows)}\n\n"

        table_content += "Sample Records:\n"
        for row in rows[:5]:
            record = dict(zip(column_names, row))
            table_content += f"{record}\n"

        doc = Document(
            page_content=table_content,
            metadata={
                'source': db_path,
                'table_name': table_name,
                'num_records': len(rows),
                'data_type': 'sql_table'
            }
        )
        documents.append(doc)

    # Strategy 2: Relationship documents from JOINs
    cursor.execute("""
        SELECT e.name, e.role, p.name as project_name, p.status
        FROM employees e
        JOIN projects p ON e.id = p.lead_id
    """)

    relationships = cursor.fetchall()
    rel_content = "Employee-Project Relationships:\n\n"
    for rel in relationships:
        rel_content += f"{rel[0]} ({rel[1]}) leads {rel[2]} - Status: {rel[3]}\n"

    rel_doc = Document(
        page_content=rel_content,
        metadata={
            'source': db_path,
            'data_type': 'sql_relationships',
            'query': 'employee_project_join'
        }
    )
    documents.append(rel_doc)

    conn.close()
    return documents
```

**Usage and output**:

```python
docs = sql_to_documents("data/databases/company.db")
# Returns 3 documents:
#   1. employees table overview (schema + 4 records)
#   2. projects table overview (schema + 4 records)
#   3. Employee-Project relationship document (from JOIN)
```

**The relationship document** is particularly important:

```
Employee-Project Relationships:

John Doe (Senior Developer) leads RAG Implementation - Status: Active
Jane Smith (Data Scientist) leads Data Pipeline - Status: Completed
Mike Johnson (Product Manager) leads Customer Portal - Status: Planning
Jane Smith (Data Scientist) leads ML Platform - Status: Active
```

This single document captures cross-table relationships that would be invisible if you only ingested individual tables. Without it, a question like "What projects does Jane Smith lead?" would require the RAG system to correlate data across two separate table documents — which embedding-based retrieval is not designed to do.

#### Two Strategies, Two Purposes

| Strategy | What It Creates | Answers Questions Like |
|----------|----------------|----------------------|
| **Table documents** | One doc per table with schema + sample rows | "What columns does the employees table have?", "How many employees are there?" |
| **Relationship documents** | JOINed data as natural language | "Who leads the RAG Implementation project?", "What projects does Jane Smith work on?" |

> **Best practice**: Use **both strategies together**. Table documents help with schema-level questions, while relationship documents handle the cross-table questions that users actually care about most.

> **Scaling tip**: For large databases with thousands of rows, don't dump the entire table into one document. Instead, create one document per row (like the CSV approach) or batch rows into groups of 10-20 records per document. Always include schema context in metadata or a prefix.

---

## Key Takeaways

1. **Everything is a `Document`** — LangChain unifies all data sources into `Document(page_content, metadata)`.
2. **Metadata is not optional** — always include `source` at minimum; add domain-specific fields for filtering.
3. **Use `RecursiveCharacterTextSplitter` by default** — it respects text hierarchy and works well for most formats.
4. **Clean PDF text after extraction** — fix whitespace, ligatures, and encoding artifacts before chunking.
5. **Choose your PDF loader based on needs**: `PyPDFLoader` for simplicity, `PyMuPDFLoader` for speed and quality.
6. **Chunk size matters** — smaller chunks (200-500 chars) for precise retrieval, larger chunks (1000-2000 chars) for more context. Always include overlap to avoid losing context at boundaries.
7. **For structured data (CSV/Excel), promote key columns to metadata** — this enables filtered retrieval (e.g., "only products in Electronics category") which dramatically improves RAG accuracy on tabular data.
8. **Custom processing beats generic loaders for production** — `CSVLoader` is great for prototyping, but formatting `page_content` as natural language and enriching metadata gives better embedding and retrieval quality.
9. **For nested JSON, flatten intelligently** — convert nested structures into readable text rather than storing raw JSON strings. Embedding models understand "Skills: Python, JavaScript" far better than `"skills": ["Python", "JavaScript"]`.
10. **For databases, create relationship documents** — JOINed data captures cross-table facts that single-table documents miss. This is the difference between answering "list all employees" and answering "who leads the most expensive project?".
11. **`create_stuff_documents_chain` + `create_retrieval_chain`** is the quickest way to build a working RAG pipeline. Use **LCEL** when you need more control over the data flow.
12. **Similarity search scores help set quality thresholds** — with L2 distance, lower is better. Filter out chunks above a score threshold to avoid injecting irrelevant context.
13. **Use `create_history_aware_retriever` for multi-turn conversations** — it reformulates follow-up questions into standalone queries so the retriever can find relevant context even when users use pronouns like "it" or "that".
14. **LLM providers are swappable** — the retriever, embeddings, and vector store are independent of the LLM. You can switch from OpenAI to Groq (or any other provider) without touching the rest of the pipeline.

---

## Building a RAG System with LangChain and ChromaDB

This section brings together everything from the data ingestion modules and walks through building a **complete, end-to-end RAG pipeline** — from loading documents all the way to generating answers grounded in your data.

### RAG Architecture Overview

RAG (Retrieval-Augmented Generation) combines the generative power of LLMs with external knowledge retrieval. The pipeline has 8 steps:

```
1. Document Loading    → Load documents from various sources
2. Document Splitting  → Break documents into smaller chunks
3. Embedding Generation → Convert chunks into vector representations
4. Vector Storage      → Store embeddings in ChromaDB
5. Query Processing    → Convert user query to embedding
6. Similarity Search   → Find relevant chunks from vector store
7. Context Augmentation → Combine retrieved chunks with query
8. Response Generation → LLM generates answer using context
```

**Why RAG over plain LLM?**

- **Reduces hallucinations** — the LLM answers based on retrieved facts, not just training data
- **Provides up-to-date information** — your vector store can be updated without retraining the model
- **Allows citing sources** — you know exactly which documents contributed to an answer
- **Works with domain-specific knowledge** — ingest proprietary docs the LLM was never trained on

### Setup and Imports

```python
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

import numpy as np
from typing import List
```

### Step 1: Prepare and Load Documents

Create sample documents about AI topics, then load them using `DirectoryLoader`:

```python
sample_docs = [
    """
    Machine Learning Fundamentals

    Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. There are three main
    types of machine learning: supervised learning, unsupervised learning, and reinforcement
    learning. Supervised learning uses labeled data to train models, while unsupervised
    learning finds patterns in unlabeled data. Reinforcement learning learns through
    interaction with an environment using rewards and penalties.
    """,

    """
    Deep Learning and Neural Networks

    Deep learning is a subset of machine learning based on artificial neural networks.
    These networks are inspired by the human brain and consist of layers of interconnected
    nodes. Deep learning has revolutionized fields like computer vision, natural language
    processing, and speech recognition. Convolutional Neural Networks (CNNs) are particularly
    effective for image processing, while Recurrent Neural Networks (RNNs) and Transformers
    excel at sequential data processing.
    """,

    """
    Natural Language Processing (NLP)

    NLP is a field of AI that focuses on the interaction between computers and human language.
    Key tasks in NLP include text classification, named entity recognition, sentiment analysis,
    machine translation, and question answering. Modern NLP heavily relies on transformer
    architectures like BERT, GPT, and T5. These models use attention mechanisms to understand
    context and relationships between words in text.
    """
]

# Save to files
for i, doc in enumerate(sample_docs):
    with open(f"data/doc_{i}.txt", "w") as f:
        f.write(doc)

# Load with DirectoryLoader
loader = DirectoryLoader(
    "data",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")  # 3
```

### Step 2: Split Documents into Chunks

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks from {len(documents)} documents")  # 5 chunks
print(f"Chunk example: {chunks[0].page_content[:150]}...")
print(f"Metadata: {chunks[0].metadata}")
```

Each chunk retains the `metadata` from its parent document (including `source`), so you can always trace a chunk back to its origin file.

### Step 3: Generate Embeddings

Embeddings convert text into high-dimensional numerical vectors where semantically similar texts are placed close together in vector space.

```python
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()  # Uses text-embedding-ada-002 by default

sample_text = "Machine Learning is fascinating"
vector = embeddings.embed_query(sample_text)
print(f"Embedding dimension: {len(vector)}")  # 1536 dimensions
print(f"First 5 values: {vector[:5]}")
```

The embedding model maps any text string to a 1536-dimensional vector. Two texts about similar topics will have vectors that are close together (small cosine distance), while unrelated texts will be far apart.

### Step 4: Store Embeddings in ChromaDB

ChromaDB is an open-source vector database that stores embeddings and supports fast similarity search.

```python
persist_directory = "./chroma_db"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory=persist_directory,
    collection_name="rag_collection"
)

print(f"Vector store created with {vectorstore._collection.count()} vectors")
print(f"Persisted to: {persist_directory}")
```

`Chroma.from_documents` does three things in one call:
1. Generates embeddings for each chunk's `page_content`
2. Stores the vectors alongside the original text and metadata
3. Persists everything to disk at `persist_directory`

### Step 5: Similarity Search

Query the vector store to find chunks most relevant to a question:

```python
query = "What are the types of machine learning?"
similar_docs = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(similar_docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content[:200] + "...")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
```

Under the hood: the query is embedded into a vector, and ChromaDB finds the `k` nearest vectors using distance metrics.

#### Similarity Search with Scores

To also get the distance scores (useful for setting relevance thresholds):

```python
results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:80]}...")
```

**Understanding scores** — ChromaDB defaults to **L2 (Euclidean) distance**:

| Metric | Lower = ? | Score of 0 | Typical Range |
|--------|-----------|------------|---------------|
| **L2 distance** (default) | More similar | Identical vectors | 0 to 2+ |
| **Cosine similarity** (if configured) | Less similar | Orthogonal | -1 to 1 (1 = identical) |

### Step 6: Initialize the LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
```

Or use `init_chat_model` for a provider-agnostic approach:

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-3.5-turbo")
# or: llm = init_chat_model("groq:gemma2-9b-it")
```

### Step 7: Build the RAG Chain

The RAG chain connects the retriever (vector store) to the LLM through a prompt template. LangChain provides two approaches.

#### Approach 1: `create_retrieval_chain` (High-Level)

```python
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# Convert vector store to retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create prompt template
system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context: {context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create document chain — "stuffs" all retrieved docs into the prompt
document_chain = create_stuff_documents_chain(llm, prompt)

# Create the full RAG chain
rag_chain = create_retrieval_chain(retriever, document_chain)
```

**What `create_stuff_documents_chain` does**: Takes retrieved documents, concatenates them, inserts them into the `{context}` placeholder in the prompt, sends the full prompt to the LLM, and returns the response. It's called "stuff" because it stuffs all documents into a single prompt.

**What `create_retrieval_chain` does**: Combines the retriever and document chain into a single pipeline. When invoked, it: (1) takes the user's input, (2) retrieves relevant documents, (3) passes them to the document chain, and (4) returns both the answer and the retrieved context.

**Querying**:

```python
response = rag_chain.invoke({"input": "What is Deep Learning?"})

print(response['answer'])
# "Deep learning is a subset of machine learning that relies on artificial neural
#  networks inspired by the human brain..."

print(response['context'])  # List of retrieved Document objects
```

The response dict contains three keys:
- `input` — the original question
- `context` — the list of `Document` objects retrieved
- `answer` — the LLM's response grounded in the context

#### Approach 2: LCEL (LangChain Expression Language) — More Flexible

LCEL gives you fine-grained control over how data flows through the chain using the pipe (`|`) operator:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

custom_prompt = ChatPromptTemplate.from_template("""Use the following context to answer the question.
If you don't know the answer based on the context, say you don't know.
Provide specific details from the context to support your answer.

Context:
{context}

Question: {question}

Answer:""")

rag_chain_lcel = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | custom_prompt
    | llm
    | StrOutputParser()
)
```

**How the LCEL chain works step by step**:

1. The input string (the question) enters the chain
2. It's sent in parallel to two branches:
   - `"context"`: the question goes to the retriever, which returns documents, then `format_docs` concatenates them into a single string
   - `"question"`: `RunnablePassthrough()` passes the question through unchanged
3. Both results fill the `custom_prompt` template
4. The filled prompt goes to the `llm`
5. `StrOutputParser()` extracts the text content from the LLM response

**Querying** — note that LCEL chains take a string directly (not a dict):

```python
response = rag_chain_lcel.invoke("What is Deep Learning?")
print(response)
# "Deep learning is a subset of machine learning based on artificial neural networks..."
```

To also get source documents separately:

```python
docs = retriever.invoke("What is Deep Learning?")
```

### Adding New Documents to an Existing Vector Store

You can incrementally add documents without rebuilding the entire store:

```python
new_document = """
Reinforcement Learning in Detail

Reinforcement learning (RL) is a type of machine learning where an agent learns to make
decisions by interacting with an environment. The agent receives rewards or penalties
based on its actions and learns to maximize cumulative reward over time. Key concepts
in RL include: states, actions, rewards, policies, and value functions. Popular RL
algorithms include Q-learning, Deep Q-Networks (DQN), Policy Gradient methods, and
Actor-Critic methods. RL has been successfully applied to game playing (like AlphaGo),
robotics, and autonomous systems.
"""

new_doc = Document(
    page_content=new_document,
    metadata={"source": "manual_addition", "topic": "reinforcement_learning"}
)

# Split, then add
new_chunks = text_splitter.split_documents([new_doc])
vectorstore.add_documents(new_chunks)

print(f"Added {len(new_chunks)} new chunks")
print(f"Total vectors now: {vectorstore._collection.count()}")
```

After adding, queries about reinforcement learning will now retrieve the new, more detailed content. The retriever automatically sees the updated store — no need to recreate the chain.

### Conversational RAG — Memory Across Turns

Standard RAG retrieves documents based only on the current query, which breaks down for follow-up questions:

```
User: "Tell me about machine learning"
Bot:  (explains ML)
User: "What are its main types?"  ← "its" refers to ML, but the retriever doesn't know that
```

The solution is a **history-aware retriever** that reformulates context-dependent questions into standalone queries before retrieval.

```python
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Prompt that reformulates questions using chat history
contextualize_q_system_prompt = """Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create the history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
```

**How it works**: Before retrieval, the LLM sees the chat history and the new question. If the question references prior context (like "its" or "that"), the LLM reformulates it into a standalone query (e.g., "What are its main types?" becomes "What are the main types of machine learning?"). The reformulated query then goes to the retriever.

**Build the full conversational chain**:

```python
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context: {context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

conversational_rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)
```

**Multi-turn conversation**:

```python
chat_history = []

# Turn 1
result1 = conversational_rag_chain.invoke({
    "chat_history": chat_history,
    "input": "What is machine learning?"
})
print(f"A: {result1['answer']}")

# Update history
chat_history.extend([
    HumanMessage(content="What is machine learning?"),
    AIMessage(content=result1['answer'])
])

# Turn 2 — follow-up with pronoun reference
result2 = conversational_rag_chain.invoke({
    "chat_history": chat_history,
    "input": "What are its main types?"  # "its" → ML, resolved via history
})
print(f"A: {result2['answer']}")
# "The main types of machine learning are supervised learning, unsupervised learning,
#  and reinforcement learning..."
```

The key insight: without `create_history_aware_retriever`, "What are its main types?" would be sent to the retriever as-is, and it wouldn't know "its" refers to machine learning. With the history-aware retriever, the question is first reformulated to "What are the main types of machine learning?" before retrieval.

### Using Alternative LLM Providers (Groq)

You can swap the LLM provider without changing the rest of the pipeline:

```python
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model

# Direct initialization
llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"))

# Or using init_chat_model (provider-agnostic)
llm = init_chat_model("groq:gemma2-9b-it")
```

After swapping the LLM, rebuild the chains with the new `llm` instance. The retriever, vector store, and embeddings stay the same — only the generation step changes.

> **Tip**: Groq provides extremely fast inference for open-source models like Gemma, Llama, and Mixtral. It's a good choice when you want low-latency RAG responses without depending on OpenAI.

---

## Key Takeaways
