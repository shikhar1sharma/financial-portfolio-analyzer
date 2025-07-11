"""
Portfolio Management MCP Server
Handles portfolio tracking, positions, and basic analytics
"""

from fastmcp import FastMCP
import sqlite3
import json
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

app = FastMCP("portfolio-management")

class PortfolioManager:
    """Manages portfolio data and operations"""
    
    def __init__(self, db_path: str = "data/portfolio.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """
        Initialize the portfolio database
        Create database schema for positions, transactions, etc.
        """
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

            # Create transactions table# Create transactions table for detailed history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    transaction_type TEXT NOT NULL, -- 'BUY' or 'SELL'
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    transaction_date TEXT NOT NULL,
                    fees REAL DEFAULT 0,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create portfolio_metadata table for overall portfolio info
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
            self.logger
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def add_position(self, symbol: str, quantity: float, purchase_price: float, 
                    purchase_date: str = None) -> Dict[str, Any]:
        """
        Add a new position to portfolio
        """
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
            cursor.execute('SELECT id FROM positions WHERE symbol = ?', (symbol,))
            existing_position = cursor.fetchone()

            if existing_position:
                # update existing position (average cost)
                old_quantity = existing_position[2]
                old_price = existing_position[3]
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
        finally:
            conn.close()

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get complete portfolio summary
        Calculate portfolio value, P&L, allocation, etc.
        """
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
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Calculate advanced portfolio metrics
        Implement Sharpe ratio, volatility, beta, etc.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all positions
            cursor.execute('SELECT symbol, quantity, purchase_price FROM positions')
            positions = cursor.fetchall()
            
            if not positions:
                return {'status': 'error', 'error': 'No positions found'}
            
            # Get historical data for all positions
            portfolio_returns = []
            market_returns = []
            
            # Get S&P 500 as market benchmark
            market_ticker = yf.Ticker('^GSPC')
            market_hist = market_ticker.history(period="1y")
            market_returns = market_hist['Close'].pct_change().dropna()
            
            # Calculate portfolio daily returns
            symbols = [pos[0] for pos in positions]
            quantities = [pos[1] for pos in positions]
            
            # Get historical data for portfolio
            portfolio_data = {}
            total_investment = sum(pos[1] * pos[2] for pos in positions)
            
            for symbol, quantity, purchase_price in positions:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1y")
                    
                    if not hist.empty:
                        weight = (quantity * purchase_price) / total_investment
                        returns = hist['Close'].pct_change().dropna()
                        portfolio_data[symbol] = {
                            'returns': returns,
                            'weight': weight
                        }
                except Exception as e:
                    self.logger.warning(f"Could not fetch data for {symbol}: {e}")
            
            if not portfolio_data:
                return {'status': 'error', 'error': 'Could not fetch historical data'}
            
            # Calculate weighted portfolio returns
            common_dates = None
            for symbol, data in portfolio_data.items():
                if common_dates is None:
                    common_dates = data['returns'].index
                else:
                    common_dates = common_dates.intersection(data['returns'].index)
            
            portfolio_returns = pd.Series(0, index=common_dates)
            for symbol, data in portfolio_data.items():
                portfolio_returns += data['returns'].loc[common_dates] * data['weight']
            
            # Align market returns with portfolio returns
            aligned_market_returns = market_returns.loc[common_dates]
            
            # Calculate metrics
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            portfolio_mean_return = portfolio_returns.mean() * 252  # Annualized
            
            # Risk-free rate (approximate)
            risk_free_rate = 0.02  # 2% annual
            
            # Sharpe Ratio
            sharpe_ratio = (portfolio_mean_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Beta calculation
            covariance = np.cov(portfolio_returns, aligned_market_returns)[0, 1]
            market_variance = np.var(aligned_market_returns)
            beta = covariance / market_variance if market_variance > 0 else 0
            
            # Maximum Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5)
            
            conn.close()
            
            return {
                'status': 'success',
                'portfolio_metrics': {
                    'annualized_return': float(portfolio_mean_return),
                    'annualized_volatility': float(portfolio_volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'beta': float(beta),
                    'max_drawdown': float(max_drawdown),
                    'value_at_risk_95': float(var_95),
                    'total_positions': len(positions),
                    'calculation_period': '1 Year',
                    'risk_free_rate': risk_free_rate
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def calculate_returns(self, timeframe: str = "1M") -> Dict[str, Any]:
        """Calculate portfolio returns for specified timeframe"""
        try:
            period_map = {
                "1D": 1,
                "1W": 7,
                "1M": 30,
                "3M": 90,
                "6M": 180,
                "1Y": 365
            }
            
            if timeframe not in period_map:
                return {'status': 'error', 'error': f'Unsupported timeframe: {timeframe}'}
            
            days = period_map[timeframe]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get portfolio summary
            portfolio_summary = self.get_portfolio_summary()
            
            if portfolio_summary['status'] != 'success':
                return portfolio_summary
            
            positions = portfolio_summary['positions']
            
            # Calculate returns for each position
            position_returns = []
            total_return = 0
            total_weight = 0
            
            for position in positions:
                symbol = position['symbol']
                current_value = position['market_value']
                
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    if len(hist) >= 2:
                        start_price = hist['Close'].iloc[0]
                        end_price = hist['Close'].iloc[-1]
                        period_return = (end_price - start_price) / start_price
                        
                        weight = current_value / portfolio_summary['portfolio_summary']['total_market_value']
                        weighted_return = period_return * weight
                        
                        position_returns.append({
                            'symbol': symbol,
                            'period_return': float(period_return * 100),
                            'weight': float(weight * 100),
                            'contribution': float(weighted_return * 100)
                        })
                        
                        total_return += weighted_return
                        total_weight += weight
                        
                except Exception as e:
                    self.logger.warning(f"Could not calculate returns for {symbol}: {e}")
            
            return {
                'status': 'success',
                'returns_analysis': {
                    'timeframe': timeframe,
                    'portfolio_return': float(total_return * 100),
                    'period_start': start_date.isoformat(),
                    'period_end': end_date.isoformat(),
                    'position_returns': position_returns
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating returns: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
    def get_asset_allocation(self) -> Dict[str, Any]:
        """Get portfolio asset allocation breakdown"""
        try:
            portfolio_summary = self.get_portfolio_summary()
            
            if portfolio_summary['status'] != 'success':
                return portfolio_summary
            
            positions = portfolio_summary['positions']
            
            # Group by sector
            sector_allocation = {}
            symbol_allocation = {}
            
            for position in positions:
                symbol = position['symbol']
                sector = position['sector']
                market_value = position['market_value']
                allocation_percent = position['allocation_percent']
                
                # By sector
                if sector not in sector_allocation:
                    sector_allocation[sector] = {
                        'market_value': 0,
                        'allocation_percent': 0,
                        'positions': []
                    }
                
                sector_allocation[sector]['market_value'] += market_value
                sector_allocation[sector]['allocation_percent'] += allocation_percent
                sector_allocation[sector]['positions'].append(symbol)
                
                # By symbol
                symbol_allocation[symbol] = {
                    'market_value': market_value,
                    'allocation_percent': allocation_percent,
                    'sector': sector
                }
            
            return {
                'status': 'success',
                'asset_allocation': {
                    'by_sector': sector_allocation,
                    'by_symbol': symbol_allocation,
                    'total_market_value': portfolio_summary['portfolio_summary']['total_market_value'],
                    'diversification_score': len(sector_allocation)  # Simple diversification measure
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating asset allocation: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

portfolio_manager = PortfolioManager()

@app.tool()
def add_stock_position(symbol: str, quantity: float, purchase_price: float, 
                      purchase_date: str = None) -> Dict[str, Any]:
    """Add a new stock position to the portfolio"""
    return portfolio_manager.add_position(symbol, quantity, purchase_price, purchase_date)

@app.tool()
def get_portfolio_overview() -> Dict[str, Any]:
    """Get complete portfolio overview and summary"""
    return portfolio_manager.get_portfolio_summary()

@app.tool()
def calculate_returns(timeframe: str = "1M") -> Dict[str, Any]:
    """Calculate portfolio returns for specified timeframe"""
    return portfolio_manager.calculate_returns(timeframe)

@app.tool()
def get_asset_allocation() -> Dict[str, Any]:
    """Get portfolio asset allocation breakdown"""
    return portfolio_manager.get_asset_allocation()

@app.tool()
def get_portfolio_metrics() -> Dict[str, Any]:
    """Calculate advanced portfolio metrics including Sharpe ratio, beta, volatility, and risk measures"""
    return portfolio_manager.calculate_portfolio_metrics()

@app.tool()
def sell_position(symbol: str, quantity: float, sale_price: float, 
                 sale_date: str = None) -> Dict[str, Any]:
    """Sell shares from an existing position
    
    Args:
        symbol: Stock ticker symbol
        quantity: Number of shares to sell
        sale_price: Sale price per share
        sale_date: Date of sale (ISO format, optional - defaults to now)
    """

    try:
        if sale_date is None:
            sale_date = datetime.now().isoformat()
        
        conn = sqlite3.connect(portfolio_manager.db_path)
        cursor = conn.cursor()
        
        # Check if position exists and has enough shares
        cursor.execute('SELECT * FROM positions WHERE symbol = ?', (symbol.upper(),))
        position = cursor.fetchone()
        
        if not position:
            return {
                'status': 'error',
                'error': f'No position found for {symbol.upper()}',
                'timestamp': datetime.now().isoformat()
            }
        
        current_quantity = position[2]
        
        if quantity > current_quantity:
            return {
                'status': 'error',
                'error': f'Insufficient shares. Current position: {current_quantity}, requested: {quantity}',
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate realized P&L
        purchase_price = position[3]
        realized_pnl = (sale_price - purchase_price) * quantity
        
        # Update position
        new_quantity = current_quantity - quantity
        
        if new_quantity == 0:
            # Remove position entirely
            cursor.execute('DELETE FROM positions WHERE symbol = ?', (symbol.upper(),))
        else:
            # Update quantity
            cursor.execute('UPDATE positions SET quantity = ? WHERE symbol = ?', 
                         (new_quantity, symbol.upper()))
        
        # Add transaction record
        cursor.execute('''
            INSERT INTO transactions (symbol, transaction_type, quantity, price, transaction_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol.upper(), 'SELL', quantity, sale_price, sale_date))
        
        conn.commit()
        conn.close()
        
        return {
            'status': 'success',
            'action': 'sold',
            'symbol': symbol.upper(),
            'quantity_sold': quantity,
            'sale_price': sale_price,
            'realized_pnl': float(realized_pnl),
            'remaining_quantity': float(new_quantity),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

        
if __name__ == "__main__":
    app.run(port=8002)
