import pino from 'pino';
import { config } from '../config/env.js';

export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error',
}

export class Logger {
  private static instance: Logger;
  private pinoLogger: pino.Logger;

  private constructor() {
    // Configure Pino logger
    const isDevelopment = config.nodeEnv === 'development';

    this.pinoLogger = pino({
      level: isDevelopment ? 'debug' : 'info',
      formatters: {
        level: (label) => {
          return { level: label };
        },
      },
      transport: isDevelopment ? {
        target: 'pino-pretty',
        options: {
          colorize: true,
          translateTime: 'SYS:standard',
          ignore: 'pid,hostname',
        },
      } : undefined,
    });
  }

  static getInstance(): Logger {
    if (!Logger.instance) {
      Logger.instance = new Logger();
    }
    return Logger.instance;
  }

  setLogLevel(level: LogLevel): void {
    this.pinoLogger.level = level;
  }

  debug(message: string, ...args: unknown[]): void {
    this.pinoLogger.debug(args.length > 0 ? { args } : {}, message);
  }

  info(message: string, ...args: unknown[]): void {
    this.pinoLogger.info(args.length > 0 ? { args } : {}, message);
  }

  warn(message: string, ...args: unknown[]): void {
    this.pinoLogger.warn(args.length > 0 ? { args } : {}, message);
  }

  error(message: string, ...args: unknown[]): void {
    this.pinoLogger.error(args.length > 0 ? { args } : {}, message);
  }

  // Additional Pino-specific methods
  child(bindings: Record<string, unknown>): Logger {
    const childLogger = new Logger();
    childLogger.pinoLogger = this.pinoLogger.child(bindings);
    return childLogger;
  }
}

export const logger = Logger.getInstance();
