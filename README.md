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
