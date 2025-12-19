# RAG Implementation Guide

## Overview
This implementation provides a complete RAG (Retrieval-Augmented Generation) system that:
1. Loads documents from the `src/RagDocs` folder
2. Splits documents into chunks
3. Creates embeddings using OpenAI's embedding model
4. Stores embeddings in ChromaDB vector database
5. Performs similarity searches on the stored documents

## Setup Instructions

### 1. Install Dependencies
```bash
pnpm install
```

### 2. Start ChromaDB
You need to have ChromaDB running. You can use Docker:

```bash
docker run -p 8000:8000 chromadb/chroma
```

Or install and run ChromaDB locally:
```bash
pip install chromadb
chroma run --host localhost --port 8000
```

### 3. Configure Environment Variables
Create a `.env` file in the project root with:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
CHROMA_URL=http://localhost:8000
NODE_ENV=development
```

### 4. Add Documents
Place your documents (`.txt` or `.pdf` files) in the `src/RagDocs/` folder.
Three sample documents have been provided to get you started.

### 5. Run the Application

Build and run:
```bash
pnpm start
```

Or use development mode with auto-reload:
```bash
pnpm dev
```

## Project Structure

```
src/
├── index.ts                    # Main entry point with RAG workflow
├── config/
│   └── env.ts                 # Environment configuration
├── services/
│   ├── documentLoader.ts      # Document loading and splitting
│   └── embeddingService.ts    # Embedding creation and vector search
├── utils/
│   └── logger.ts              # Logging utility
└── RagDocs/                   # Place your documents here
    ├── sample1.txt            # Sample document about AI/ML
    ├── sample2.txt            # Sample document about LLMs
    └── sample3.txt            # Sample document about RAG
```

## Services

### DocumentLoaderService
- **loadDocuments()**: Loads all .txt and .pdf files from RagDocs
- **splitDocuments()**: Splits documents into chunks (1000 chars, 200 overlap)
- **loadAndSplitDocuments()**: Convenience method that does both

### EmbeddingService
- **createEmbeddings()**: Creates embeddings and stores in ChromaDB
- **loadVectorStore()**: Loads existing vector store from ChromaDB
- **similaritySearch()**: Finds similar documents for a query
- **similaritySearchWithScore()**: Returns documents with similarity scores
- **addDocuments()**: Adds more documents to existing vector store

## Usage Examples

### Basic RAG Flow (Already in index.ts)
```typescript
import { DocumentLoaderService } from './services/documentLoader.js';
import { EmbeddingService } from './services/embeddingService.js';

// Load documents
const documentLoader = new DocumentLoaderService();
const documents = await documentLoader.loadAndSplitDocuments();

// Create embeddings
const embeddingService = new EmbeddingService('rag_documents');
const vectorStore = await embeddingService.createEmbeddings(documents);

// Search
const results = await embeddingService.similaritySearch(
  'What is machine learning?',
  vectorStore,
  3
);
```

### Load Existing Vector Store
```typescript
const embeddingService = new EmbeddingService('rag_documents');
const vectorStore = await embeddingService.loadVectorStore();
const results = await embeddingService.similaritySearch(
  'Explain RAG',
  vectorStore
);
```

### Add More Documents Later
```typescript
const documentLoader = new DocumentLoaderService();
const newDocuments = await documentLoader.loadAndSplitDocuments();

const embeddingService = new EmbeddingService('rag_documents');
const vectorStore = await embeddingService.loadVectorStore();

await embeddingService.addDocuments(newDocuments, vectorStore);
```

## Configuration Options

### Text Splitting
Modify chunk size and overlap in `documentLoader.ts`:
```typescript
this.textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,      // Adjust chunk size
  chunkOverlap: 200,    // Adjust overlap
});
```

### Embedding Model
Change the embedding model in `.env`:
```env
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# or text-embedding-3-large for better quality
```

### ChromaDB Collection
Change collection name when initializing:
```typescript
const embeddingService = new EmbeddingService('my_custom_collection');
```

## Troubleshooting

### ChromaDB Connection Error
- Ensure ChromaDB is running on the configured URL
- Check if port 8000 is available
- Verify CHROMA_URL in .env file

### OpenAI API Error
- Verify your API key is correct
- Check if you have sufficient API credits
- Ensure the embedding model name is valid

### No Documents Found
- Add .txt or .pdf files to `src/RagDocs/`
- Check file permissions
- Verify the path in DocumentLoaderService constructor

## Next Steps

1. Integrate with a chat interface
2. Add support for more document types (Word, HTML, etc.)
3. Implement a query interface with conversation history
4. Add caching for better performance
5. Implement hybrid search (combining dense and sparse retrieval)
6. Add metadata filtering capabilities

## Resources

- [LangChain Documentation](https://js.langchain.com/docs/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

