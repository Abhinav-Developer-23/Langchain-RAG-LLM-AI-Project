import { OpenAIEmbeddings } from '@langchain/openai';
import { Chroma } from '@langchain/community/vectorstores/chroma';
import { Document } from '@langchain/core/documents';
import { config } from '../config/env.js';
import { logger } from '../utils/logger.js';

export class EmbeddingService {
  private embeddings: OpenAIEmbeddings;
  private collectionName: string;

  constructor(collectionName: string = 'rag_documents') {
    this.collectionName = collectionName;
    
    // Initialize OpenAI Embeddings
    this.embeddings = new OpenAIEmbeddings({
      openAIApiKey: config.openai.apiKey,
      modelName: config.openai.embeddingModel || 'text-embedding-ada-002',
    });

    logger.info('Embedding service initialized with OpenAI');
  }

  /**
   * Create embeddings for documents and store in ChromaDB
   */
  async createEmbeddings(documents: Document[]): Promise<Chroma> {
    try {
      logger.info(`Creating embeddings for ${documents.length} documents`);

      const vectorStore = await Chroma.fromDocuments(
        documents,
        this.embeddings,
        {
          collectionName: this.collectionName,
          url: config.chroma.url,
        }
      );

      logger.info(`Successfully created embeddings and stored in ChromaDB collection: ${this.collectionName}`);
      return vectorStore;
    } catch (error) {
      logger.error('Error creating embeddings:', error instanceof Error ? error.message : error);
      if (error instanceof Error && error.stack) {
        logger.error('Stack trace:', error.stack);
      }
      throw error;
    }
  }

  /**
   * Load existing vector store from ChromaDB
   */
  async loadVectorStore(): Promise<Chroma> {
    try {
      logger.info(`Loading vector store from collection: ${this.collectionName}`);

      const vectorStore = await Chroma.fromExistingCollection(
        this.embeddings,
        {
          collectionName: this.collectionName,
          url: config.chroma.url,
        }
      );

      logger.info('Vector store loaded successfully');
      return vectorStore;
    } catch (error) {
      logger.error('Error loading vector store:', error);
      throw error;
    }
  }

  /**
   * Perform similarity search
   */
  async similaritySearch(
    query: string,
    vectorStore: Chroma,
    k: number = 4
  ): Promise<Document[]> {
    try {
      logger.info(`Performing similarity search for: "${query}"`);

      const results = await vectorStore.similaritySearch(query, k);
      
      logger.info(`Found ${results.length} similar documents`);
      return results;
    } catch (error) {
      logger.error('Error performing similarity search:', error);
      throw error;
    }
  }

  /**
   * Perform similarity search with scores
   */
  async similaritySearchWithScore(
    query: string,
    vectorStore: Chroma,
    k: number = 4
  ): Promise<[Document, number][]> {
    try {
      logger.info(`Performing similarity search with scores for: "${query}"`);

      const results = await vectorStore.similaritySearchWithScore(query, k);
      
      logger.info(`Found ${results.length} similar documents with scores`);
      return results;
    } catch (error) {
      logger.error('Error performing similarity search with scores:', error);
      throw error;
    }
  }

  /**
   * Add more documents to existing vector store
   */
  async addDocuments(documents: Document[], vectorStore: Chroma): Promise<void> {
    try {
      logger.info(`Adding ${documents.length} documents to vector store`);

      await vectorStore.addDocuments(documents);

      logger.info('Documents added successfully');
    } catch (error) {
      logger.error('Error adding documents:', error);
      throw error;
    }
  }
}

