import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

export const config = {
  openai: {
    apiKey: process.env.OPENAI_API_KEY,
    modelName: process.env.OPENAI_MODEL_NAME,
    embeddingModel: process.env.OPENAI_EMBEDDING_MODEL,
  },
  chroma: {
    url: process.env.CHROMA_URL || 'http://localhost:8000',
  },
  nodeEnv: process.env.NODE_ENV || 'development',
};

/**
 * Validates that required environment variables are set and not empty
 * Throws error and exits if any required variables are missing or empty
 */
export function validateEnvironment(): void {
  const requiredVars = [
    'OPENAI_API_KEY',
    'OPENAI_MODEL_NAME',
    'OPENAI_EMBEDDING_MODEL',
  ];

  // Check for missing or empty variables
  const missing = requiredVars.filter(
    varName => !process.env[varName] || process.env[varName]?.trim() === ''
  );

  if (missing.length > 0) {
    console.error('\x1b[31m%s\x1b[0m', '❌ CONFIGURATION ERROR:');
    console.error(
      `\nMissing or empty required environment variables:\n  - ${missing.join('\n  - ')}\n`
    );
    console.error('Please set these in your .env file with valid values.\n');
    console.error('Example .env file:');
    console.error('  OPENAI_API_KEY=sk-...');
    console.error('  OPENAI_MODEL_NAME=gpt-4');
    console.error('  OPENAI_EMBEDDING_MODEL=text-embedding-ada-002');
    console.error('  CHROMA_URL=http://localhost:8000');
    console.error('  NODE_ENV=development\n');
    
    process.exit(1); // Exit the program with error code
  }

  // Validate ChromaDB URL format if provided
  if (process.env.CHROMA_URL) {
    try {
      new URL(process.env.CHROMA_URL);
    } catch (error) {
      console.error('\x1b[31m%s\x1b[0m', '❌ CONFIGURATION ERROR:');
      console.error(`\nInvalid CHROMA_URL format: ${process.env.CHROMA_URL}`);
      console.error('Please provide a valid URL (e.g., http://localhost:8000)\n');
      process.exit(1);
    }
  }
}
