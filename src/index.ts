import { DocumentLoaderService } from './services/documentLoader.js';
import { EmbeddingService } from './services/embeddingService.js';
import { validateEnvironment, config } from './config/env.js';
import { logger } from './utils/logger.js';

async function main() {
  try {
    // Validate environment variables
    validateEnvironment();
    logger.info('Environment validated successfully');

    // Initialize services
    const documentLoader = new DocumentLoaderService();
    const embeddingService = new EmbeddingService('rag_documents');

    logger.info('=== Starting RAG Document Processing ===');

    // Step 1: Load and split documents from RagDocs
    logger.info('Step 1: Loading documents from RagDocs folder...');
    const documents = await documentLoader.loadAndSplitDocuments();

    if (documents.length === 0) {
      logger.warn('No documents found in RagDocs folder. Please add some .txt or .pdf files to the src/RagDocs directory.');
      return;
    }

    // Step 2: Create embeddings and store in ChromaDB
    logger.info('Step 2: Creating embeddings and storing in ChromaDB...');
    const vectorStore = await embeddingService.createEmbeddings(documents);

    logger.info('Step 3: Embeddings created successfully!');

    // Example: Perform a similarity search (optional demo)
    logger.info('=== Demonstrating Similarity Search ===');
    const sampleQuery = 'What is this document about?';
    const similarDocs = await embeddingService.similaritySearch(
      sampleQuery,
      vectorStore,
      3
    );

    logger.info(`\nSearch Results for: "${sampleQuery}"\n`);
    similarDocs.forEach((doc, index) => {
      logger.info(`Result ${index + 1}:`);
      logger.info(`Content: ${doc.pageContent.substring(0, 200)}...`);
      logger.info(`Source: ${doc.metadata.source || 'Unknown'}\n`);
    });

    // Example: Similarity search with scores
    logger.info('=== Similarity Search with Scores ===');
    const resultsWithScores = await embeddingService.similaritySearchWithScore(
      sampleQuery,
      vectorStore,
      3
    );

    logger.info(`\nSearch Results with Scores for: "${sampleQuery}"\n`);
    resultsWithScores.forEach(([doc, score], index) => {
      logger.info(`Result ${index + 1} (Score: ${score.toFixed(4)}):`);
      logger.info(`Content: ${doc.pageContent.substring(0, 200)}...`);
      logger.info(`Source: ${doc.metadata.source || 'Unknown'}\n`);
    });

    logger.info('=== RAG Processing Complete ===');
    logger.info(`Total documents processed: ${documents.length}`);
    logger.info(`Collection name: rag_documents`);
    logger.info(`ChromaDB URL: ${config.chroma.url}`);

  } catch (error) {
    logger.error('Error in main process:', error);
    process.exit(1);
  }
}

// Run the main function
main();

