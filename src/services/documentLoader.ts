import { DirectoryLoader } from '@langchain/classic/document_loaders/fs/directory';
import { TextLoader } from '@langchain/classic/document_loaders/fs/text';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/classic/text_splitter';
import { Document } from '@langchain/core/documents';
import { logger } from '../utils/logger.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class DocumentLoaderService {
  private documentsPath: string;
  private textSplitter: RecursiveCharacterTextSplitter;

  constructor(documentsPath?: string) {
    // Default to src/RagDocs in both development and production
    if (!documentsPath) {
      // Get the project root directory
      const projectRoot = path.resolve(__dirname, '../../');
      this.documentsPath = path.join(projectRoot, 'src/RagDocs');
    } else {
      this.documentsPath = documentsPath;
    }
    
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
  }

  /**
   * Load documents from the RagDocs directory
   */
  async loadDocuments(): Promise<Document[]> {
    try {
      logger.info(`Loading documents from: ${this.documentsPath}`);

      const loader = new DirectoryLoader(
        this.documentsPath,
        {
          '.txt': (filePath: string) => new TextLoader(filePath),
          '.pdf': (filePath: string) => new PDFLoader(filePath),
        }
      );

      const docs = await loader.load();
      logger.info(`Loaded ${docs.length} documents`);

      return docs;
    } catch (error) {
      logger.error('Error loading documents:', error);
      throw error;
    }
  }

  /**
   * Split documents into smaller chunks for better embedding
   */
  async splitDocuments(documents: Document[]): Promise<Document[]> {
    try {
      logger.info(`Splitting ${documents.length} documents into chunks`);
      const splitDocs = await this.textSplitter.splitDocuments(documents);
      logger.info(`Created ${splitDocs.length} chunks from documents`);
      return splitDocs;
    } catch (error) {
      logger.error('Error splitting documents:', error);
      throw error;
    }
  }

  /**
   * Load and split documents in one step
   */
  async loadAndSplitDocuments(): Promise<Document[]> {
    const documents = await this.loadDocuments();
    if (documents.length === 0) {
      logger.warn('No documents found in RagDocs directory');
      return [];
    }
    return await this.splitDocuments(documents);
  }
}

