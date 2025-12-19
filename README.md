# Langchain RAG LLM AI Project

A TypeScript-based RAG (Retrieval-Augmented Generation) application using Langchain, OpenAI, and ChromaDB.

## Features

- **TypeScript**: Full type safety with modern TypeScript configuration
- **Langchain**: Core Langchain library for LLM orchestration
- **OpenAI Integration**: GPT-3.5-turbo and text embeddings
- **ChromaDB**: Vector database for document storage and retrieval
- **ESLint**: Code linting and error checking
- **RAG Pipeline**: Ready for implementing retrieval-augmented generation workflows

## Prerequisites

- Node.js 18+ and pnpm
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Langchain-RAG-LLM-AI-Project
```

2. Install dependencies:
```bash
pnpm install
```

3. Set up environment variables:
```bash
# A .env file has been created for you. Edit it and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

## Configuration

A `.env` file has been created for you with the following variables. Make sure to replace the placeholder values:

```env
OPENAI_API_KEY=your_openai_api_key_here  # ← Replace with your actual OpenAI API key
OPENAI_MODEL_NAME=gpt-3.5-turbo          # Optional: LLM model (default: gpt-3.5-turbo)
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002 # Optional: Embedding model
CHROMA_URL=http://localhost:8000        # Default ChromaDB URL (localhost)
NODE_ENV=development                     # Development environment
```

**Important:** The application will fail to initialize LangChain services if you don't provide a valid OpenAI API key.

## Logging

The application uses **Pino** for high-performance logging:

- **Development**: Pretty-printed, colored output with timestamps
- **Production**: JSON structured logging for better parsing and monitoring
- **Log Levels**: debug, info, warn, error
- **Child Loggers**: Create contextual loggers with additional metadata

Example output in development:
```
[2025-12-19 19:13:47.537 +0000] INFO: Starting Langchain RAG LLM AI Project...
[2025-12-19 19:13:47.537 +0000] INFO: Initializing LangChain service...
```

## Usage

### Development

Run in development mode with auto-restart on file changes (using nodemon):
```bash
pnpm run dev
```

### Building

Build the TypeScript project:
```bash
pnpm run build
```

### Production

Run the production version (automatically builds first):
```bash
pnpm start
```

Or build separately and run:
```bash
pnpm run build
pnpm start
```

### Linting

Check for linting errors:
```bash
pnpm run lint
```

Auto-fix linting issues:
```bash
pnpm run lint:fix
```

## Project Structure

```
src/
├── config/
│   └── env.ts          # Environment configuration and validation
├── services/
│   └── langchain.ts    # LangChain service initialization
├── utils/
│   └── logger.ts       # Logging utility
└── index.ts            # Main application entry point
```

## Key Components

- **Environment Validation**: Ensures all required API keys and configuration are present
- **LangChain Service**: Manages OpenAI models, embeddings, and vector store connections
- **Pino Logger**: High-performance JSON logging with pretty printing in development
- **TypeScript**: Full type safety throughout the application

## Next Steps

The project is set up with the basic infrastructure. To implement your RAG pipeline:

1. Add document loading and processing (PDF, text files, etc.)
2. Implement vector store operations (indexing, similarity search)
3. Create question-answering chains
4. Add a chat interface or API endpoints

## Dependencies

- `langchain`: Core Langchain library
- `@langchain/openai`: OpenAI integrations
- `@langchain/community`: Community integrations
- `chromadb`: Vector database
- `pdf-parse`: PDF processing
- `dotenv`: Environment variable management
- `pino`: High-performance JSON logger
- `typescript`: TypeScript compiler
- `eslint`: Code linting

## License

ISC
