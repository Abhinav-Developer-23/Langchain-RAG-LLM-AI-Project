import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

export const config = {
  openai: {
    apiKey: process.env.OPENAI_API_KEY,
  },
  chroma: {
    url: process.env.CHROMA_URL || 'http://localhost:8000',
  },
  nodeEnv: process.env.NODE_ENV || 'development',
};

/**
 * Validates that required environment variables are set
 */
export function validateEnvironment(): void {
  const requiredVars = ['OPENAI_API_KEY'];

  const missing = requiredVars.filter(varName => !process.env[varName]);

  if (missing.length > 0) {
    throw new Error(
      `Missing required environment variables: ${missing.join(', ')}\n` +
      'Please set these in your .env file or environment.'
    );
  }
}
