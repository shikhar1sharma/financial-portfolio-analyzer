# Fixed Portfolio Server - HTTP version
# File: mcp_servers/portfolio_http_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import json
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import uvicorn

app = FastAPI(title="Portfolio Management Server", version="1.0.0")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioManager:
    """Manages portfolio data and operations"""
    
    def __init__(self, db_path: str = "data/portfolio.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize the portfolio database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    purchase_price REAL NOT NULL,
                    purchase_date TEXT,
                    current_price REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sector TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create transactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    transaction_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    transaction_date TEXT NOT NULL,
                    fees REAL DEFAULT 0,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create portfolio_metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_name TEXT DEFAULT 'Default Portfolio',
                    initial_capital REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT
                )
            ''')    

            # Insert default metadata if not exists
            cursor.execute('SELECT COUNT(*) FROM portfolio_metadata')
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                    INSERT INTO portfolio_metadata (portfolio_name, initial_capital, updated_at)
                    VALUES (?, ?, ?)
                ''', ('Default Portfolio', 10000.0, datetime.now().isoformat()))

            conn.commit()
            conn.close()
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def add_position(self, symbol: str, quantity: float, purchase_price: float, 
                    purchase_date: str = None) -> Dict[str, Any]:
        """Add a new position to portfolio"""
        try:
            if purchase_date is None:
                purchase_date = datetime.now().isoformat()

            # Get current market data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")

            current_price = hist['Close'].iloc[-1] if not hist.empty else 0.0
            sector = info.get('sector', 'Unknown')

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if position already exists
            cursor.execute('SELECT id, quantity, purchase_price FROM positions WHERE symbol = ?', (symbol.upper(),))
            existing_position = cursor.fetchone()

            if existing_position:
                # Update existing position (average cost)
                old_quantity = existing_position[1]
                old_price = existing_position[2]
                new_quantity = old_quantity + quantity
                new_avg_price = ((old_quantity * old_price) + (quantity * purchase_price)) / new_quantity
                
                cursor.execute('''
                    UPDATE positions 
                    SET quantity = ?, purchase_price = ?, current_price = ?, 
                        last_updated = ?, sector = ?
                    WHERE symbol = ?
                ''', (new_quantity, new_avg_price, current_price, 
                      datetime.now().isoformat(), sector, symbol.upper()))
                
                action = "updated"
            else:
                # Add new position
                cursor.execute('''
                    INSERT INTO positions (symbol, quantity, purchase_price, purchase_date, 
                                         current_price, last_updated, sector)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (symbol.upper(), quantity, purchase_price, purchase_date, 
                      current_price, datetime.now().isoformat(), sector))
                
                action = "added"
            
            # Add transaction record
            cursor.execute('''
                INSERT INTO transactions (symbol, transaction_type, quantity, price, transaction_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol.upper(), 'BUY', quantity, purchase_price, purchase_date))
            
            conn.commit()
            conn.close()

            return {
                'status': 'success',
                'action': action,
                'symbol': symbol.upper(),
                'quantity': quantity,
                'purchase_price': purchase_price,
                'current_price': float(current_price),
                'market_value': float(current_price * quantity),
                'unrealized_pnl': float((current_price - purchase_price) * quantity),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error adding position {symbol}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'symbol': symbol.upper(),
                'timestamp': datetime.now().isoformat()
            }

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get complete portfolio summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all positions
            cursor.execute('SELECT * FROM positions')
            positions = cursor.fetchall()
            
            if not positions:
                return {
                    'status': 'success',
                    'message': 'No positions found',
                    'total_value': 0,
                    'total_cost': 0,
                    'total_pnl': 0,
                    'positions': [],
                    'timestamp': datetime.now().isoformat()
                }
            
            portfolio_data = []
            total_market_value = 0
            total_cost_basis = 0
            
            for position in positions:
                symbol = position[1]
                quantity = position[2]
                purchase_price = position[3]
                
                # Get current price
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    current_price = hist['Close'].iloc[-1] if not hist.empty else purchase_price
                    
                    # Update current price in database
                    cursor.execute('''
                        UPDATE positions SET current_price = ?, last_updated = ? 
                        WHERE symbol = ?
                    ''', (current_price, datetime.now().isoformat(), symbol))
                    
                except Exception as e:
                    current_price = purchase_price
                    self.logger.warning(f"Could not fetch current price for {symbol}: {e}")
                
                cost_basis = quantity * purchase_price
                market_value = quantity * current_price
                unrealized_pnl = market_value - cost_basis
                pnl_percent = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
                
                position_data = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'purchase_price': purchase_price,
                    'current_price': float(current_price),
                    'cost_basis': cost_basis,
                    'market_value': float(market_value),
                    'unrealized_pnl': float(unrealized_pnl),
                    'pnl_percent': float(pnl_percent),
                    'sector': position[7] if len(position) > 7 else 'Unknown'
                }
                
                portfolio_data.append(position_data)
                total_market_value += market_value
                total_cost_basis += cost_basis
            
            conn.commit()
            conn.close()
            
            total_pnl = total_market_value - total_cost_basis
            total_return_percent = (total_pnl / total_cost_basis) * 100 if total_cost_basis > 0 else 0
            
            # Calculate allocation percentages
            for position in portfolio_data:
                position['allocation_percent'] = (position['market_value'] / total_market_value) * 100 if total_market_value > 0 else 0
            
            return {
                'status': 'success',
                'portfolio_summary': {
                    'total_market_value': float(total_market_value),
                    'total_cost_basis': float(total_cost_basis),
                    'total_unrealized_pnl': float(total_pnl),
                    'total_return_percent': float(total_return_percent),
                    'number_of_positions': len(positions),
                    'last_updated': datetime.now().isoformat()
                },
                'positions': portfolio_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    # Add other methods from the original portfolio server...
    # (calculate_portfolio_metrics, calculate_returns, get_asset_allocation, sell_position)
    # I'll include them in the next part to keep this response manageable

# Initialize the portfolio manager
portfolio_manager = PortfolioManager()

# Pydantic models
class AddPositionRequest(BaseModel):
    symbol: str
    quantity: float
    purchase_price: float
    purchase_date: Optional[str] = None

class SellPositionRequest(BaseModel):
    symbol: str
    quantity: float
    sale_price: float
    sale_date: Optional[str] = None

class TimeframeRequest(BaseModel):
    timeframe: str = "1M"

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Portfolio Management Server", "status": "running", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/tools/add_stock_position")
async def add_stock_position(request: AddPositionRequest):
    """Add a new stock position to the portfolio"""
    return portfolio_manager.add_position(
        request.symbol, 
        request.quantity, 
        request.purchase_price, 
        request.purchase_date
    )

@app.get("/tools/get_portfolio_overview")
async def get_portfolio_overview():
    """Get complete portfolio overview and summary"""
    return portfolio_manager.get_portfolio_summary()

@app.post("/tools/calculate_returns")
async def calculate_returns(request: TimeframeRequest):
    """Calculate portfolio returns for specified timeframe"""
    # Implementation would go here - simplified for now
    return {"message": "Returns calculation endpoint", "timeframe": request.timeframe}

@app.get("/tools/get_asset_allocation")
async def get_asset_allocation():
    """Get portfolio asset allocation breakdown"""
    # Implementation would go here - simplified for now
    return {"message": "Asset allocation endpoint"}

@app.get("/tools/get_portfolio_metrics")
async def get_portfolio_metrics():
    """Calculate advanced portfolio metrics"""
    # Implementation would go here - simplified for now
    return {"message": "Portfolio metrics endpoint"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)