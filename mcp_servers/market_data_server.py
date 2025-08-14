# Fixed Market Data Server - HTTP version
# File: mcp_servers/market_data_http_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import uvicorn

app = FastAPI(title="Market Data Server", version="1.0.0")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataProvider:
    """Handles all market data operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Major market indices
        self.indices = {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "DOW": "^DJI",
            "Russell 2000": "^RUT",
            "FTSE 100": "^FTSE",
            "VIX": "^VIX"
        }
    
    def get_stock_data(self, symbol: str, period: str = "1d") -> Dict[str, Any]:
        """Get stock data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info

            current_price = hist['Close'].iloc[-1] if not hist.empty else None

            if len(hist) > 1:
                price_change = current_price - hist['Close'].iloc[-2]
                percent_change_pct = (price_change / hist['Close'].iloc[-2]) * 100
            else:
                price_change = 0
                percent_change_pct = 0

            return {
                "symbol": symbol.upper(),
                "current_price": float(current_price) if current_price is not None else None,
                "price_change": float(price_change) if price_change is not None else None,
                "percent_change_pct": float(percent_change_pct), 
                "volume": int(hist['Volume'].iloc[-1]) if not hist.empty else 0,
                "high_52w": float(info.get('fiftyTwoWeekHigh', 0)),
                "low_52w": float(info.get('fiftyTwoWeekLow', 0)),
                "market_cap": float(info.get('marketCap', 0)),
                "pe_ratio": float(info.get('trailingPE', 0)),
                "company_name": info.get('longName', symbol),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "dividend_yield": float(info.get('dividendYield', 0)),
                "beta": float(info.get('beta', 0)),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return {
                "symbol": symbol.upper(),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_market_indices(self) -> Dict[str, Any]:
        """Get major market indices"""
        try:
            indices_data = {}

            for name, symbol in self.indices.items():
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")

                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    if len(hist) > 1:
                        prev_close = hist['Close'].iloc[-2]
                        change = current_price - prev_close
                        change_pct = (change / prev_close) * 100
                    else:
                        change = 0
                        change_pct = 0

                    indices_data[name] = {
                        "symbol": symbol,
                        "current_price": float(current_price),
                        "change": float(change),
                        "change_percent": float(change_pct),
                        "volume": int(hist['Volume'].iloc[-1]),
                        "timestamp": datetime.now().isoformat()
                    }
                
            return {
                "indices": indices_data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error fetching market indices: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def calculate_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Calculate technical indicators"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")

            if hist.empty:
                return {"symbol": symbol.upper(), "error": "No historical data available"}

            # Moving Averages
            ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            ma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]

            # RSI Calculation
            rsi = self._calculate_rsi(hist['Close'])
            
            # MACD Calculation
            macd_line, macd_signal, macd_histogram = self._calculate_macd(hist['Close'])

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(hist['Close'])

            # Support and Resistance levels 
            support_level = hist['Low'].rolling(window=20).min().iloc[-1]
            resistance_level = hist['High'].rolling(window=20).max().iloc[-1]

            current_price = hist['Close'].iloc[-1]

            return {
                "symbol": symbol.upper(),
                "current_price": float(current_price),
                "moving_averages": {
                    "ma_20": float(ma_20) if not pd.isna(ma_20) else None,
                    "ma_50": float(ma_50) if not pd.isna(ma_50) else None,
                    "ma_200": float(ma_200) if ma_200 and not pd.isna(ma_200) else None
                },
                "rsi": float(rsi) if not pd.isna(rsi) else None,
                "macd": {
                    "macd_line": float(macd_line) if not pd.isna(macd_line) else None,
                    "signal_line": float(macd_signal) if not pd.isna(macd_signal) else None,
                    "histogram": float(macd_histogram) if not pd.isna(macd_histogram) else None
                },
                "bollinger_bands": {
                    "upper": float(bb_upper) if not pd.isna(bb_upper) else None,
                    "middle": float(bb_middle) if not pd.isna(bb_middle) else None,
                    "lower": float(bb_lower) if not pd.isna(bb_lower) else None
                },
                "support_resistance": {
                    "support": float(support_level),
                    "resistance": float(resistance_level)
                },
                "signals": self._generate_trading_signals(current_price, ma_20, ma_50, rsi, macd_line, macd_signal),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return {
                "symbol": symbol.upper(),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index (RSI)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else None
    
    def _calculate_macd(self, prices: pd.Series, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> tuple:
        """Calculate MACD"""
        exp1 = prices.ewm(span=short_window, adjust=False).mean()
        exp2 = prices.ewm(span=long_window, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_line.iloc[-1], signal_line.iloc[-1], macd_histogram.iloc[-1]
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std_dev: int = 2) -> tuple:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)
        return upper_band.iloc[-1], rolling_mean.iloc[-1], lower_band.iloc[-1]
    
    def _generate_trading_signals(self, current_price: float, ma_20: float, ma_50: float, rsi: float, macd_line: float, macd_signal: float) -> List[str]:
        """Generate trading signals"""
        signals = []
        
        # RSI signals
        if rsi and rsi > 70:
            signals.append("RSI Overbought - Consider selling")
        elif rsi and rsi < 30:
            signals.append("RSI Oversold - Consider buying")
        
        # Moving average signals
        if ma_20 and ma_50 and ma_20 > ma_50:
            signals.append("Bullish - MA20 above MA50")
        elif ma_20 and ma_50 and ma_20 < ma_50:
            signals.append("Bearish - MA20 below MA50")
        
        # MACD signals
        if macd_line and macd_signal and macd_line > macd_signal:
            signals.append("MACD Bullish - MACD above signal line")
        elif macd_line and macd_signal and macd_line < macd_signal:
            signals.append("MACD Bearish - MACD below signal line")
        
        return signals

# Initialize the market data provider
market_provider = MarketDataProvider()

# Pydantic models for request validation
class StockRequest(BaseModel):
    symbol: str
    period: str = "1d"

class PortfolioRequest(BaseModel):
    symbols: List[str]
    weights: List[float]

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Market Data Server", "status": "running", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/tools/get_stock_price")
async def get_stock_price(request: StockRequest):
    """Get current stock price and basic information"""
    return market_provider.get_stock_data(request.symbol, request.period)

@app.post("/tools/get_portfolio_performance")
async def get_portfolio_performance(request: PortfolioRequest):
    """Calculate portfolio performance metrics"""
    try:
        symbols = request.symbols
        weights = request.weights
        
        if len(symbols) != len(weights):
            raise HTTPException(status_code=400, detail='Number of symbols must match number of weights')
        
        if abs(sum(weights) - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail='Weights must sum to 1.0')
        
        portfolio_data = []
        total_value = 0
        total_change = 0
        
        for symbol, weight in zip(symbols, weights):
            stock_data = market_provider.get_stock_data(symbol, "1mo")
            
            if 'error' not in stock_data and stock_data['current_price']:
                weighted_value = stock_data['current_price'] * weight
                weighted_change = stock_data['percent_change_pct'] * weight
                
                portfolio_data.append({
                    'symbol': symbol,
                    'weight': weight,
                    'current_price': stock_data['current_price'],
                    'weighted_contribution': weighted_change
                })
                
                total_value += weighted_value
                total_change += weighted_change
        
        return {
            'portfolio_summary': {
                'total_weighted_value': total_value,
                'total_change_percent': total_change,
                'number_of_holdings': len(symbols)
            },
            'holdings': portfolio_data,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools/get_market_overview")
async def get_market_overview():
    """Get overall market overview and indices"""
    return market_provider.get_market_indices()

@app.post("/tools/get_technical_analysis")
async def get_technical_analysis(request: StockRequest):
    """Get technical analysis for a stock"""
    return market_provider.calculate_technical_indicators(request.symbol)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)