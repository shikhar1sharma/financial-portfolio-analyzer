# AI Financial Portfolio Analyzer

A cutting-edge financial portfolio analyzer using MCP (Model Context Protocol) architecture with AI-powered insights.

## Features

- 📊 Real-time portfolio tracking
- 🤖 AI-powered stock predictions
- 📰 News sentiment analysis
- ⚡ Real-time alerts
- 📈 Advanced risk analysis
- 📋 Automated report generation

## Quick Start

In Progress...

<!--
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Start all services:
   ```bash
   python scripts/start_all_services.py
   ```

4. Access dashboard:
   ```
   http://localhost:7860
   ```
   -->

## Architecture

Built using MCP (Model Context Protocol) with multiple specialized servers:
- Market Data Server
- Portfolio Management Server  
- News Sentiment AI Server
- Stock Prediction Server
- Risk Analysis Server
- Alert System Server
- Report Generator Server

## Development

Each MCP server is a standalone service that provides specific tools.
AI agents coordinate between servers to provide comprehensive analysis.

## RAG Architecture

- We are using PostgreSQL with pgvector.
- OpenAI Embeddings (API)
   - Model: text-embedding-3-small
   - Cost: ~$0.02 per 1M tokens

### Implementation Phases

- Phase 1: Foundation
   - Set up PostgreSQL with pgvector
   - Implement basic document ingestion pipeline
   - Deploy sentence transformers for embeddings

- Phase 2: Content Integration
   - Process 5-10 key trading/finance books
   - Integrate RAG into your financial agent
