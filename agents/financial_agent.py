"""
Main Financial AI Agent
Coordinates between MCP servers to provide comprehensive analysis
"""

from smolagents import Agent
import json
from typing import Dict, Any, List
import logging
import requests

class FinancialAgent:
    """Main AI agent for financial analysis and coordination"""
    
    def __init__(self, model: str = "gpt-4"):
        self.logger = logging.getLogger(__name__)
        self.mcp_config = self.setup_mcp_config()
        try: 
            self.agent = Agent(model=model, mcp_servers=self.mcp_config)
        except Exception as e:
            self.logger.warning(f"Could not initialize agent with MCP servers: {e}")
            # Fallback to basic agent
            self.agent = Agent(model=model)

        # MCP servers endpoints
        self.portfolio_server = "http://localhost:8002"
        self.market_data_server = "http://localhost:8001"
    
    def setup_mcp_config(self) -> Dict[str, Any]:
        """
        Configure MCP servers for the agent
        """
        config = {
            "servers": [
                {
                    "name": "portfolio-management",
                    "url": "http://localhost:8002",
                    "description": "MCP server for portfolio management - handles portfolio tracking, positions, transactions, and analytics",
                    "tools": [
                        "add_stock_position",
                        "get_portfolio_overview", 
                        "calculate_returns",
                        "get_asset_allocation",
                        "get_portfolio_metrics",
                        "sell_position"
                    ]
                },
                {
                    "name": "market-data",
                    "url": "http://localhost:8001",
                    "description": "MCP server for market data",
                    "tools": [
                        "get_stock_price",
                        "get_portfolio_performance", 
                        "get_market_overview",
                        "get_technical_analysis"
                    ]
                }
            ]
        }
        return config
    
    def call_mcp_tool(self, server_url: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a tool on the specified MCP server
        """
        try:
            response = requests.post(
                f"{server_url}/tools/{tool_name}",
                json=kwargs,
                headers={"Content-Type": "application/json"}
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Error calling MCP tool {tool_name} on {server_url}: {e}")
            return {"error": str(e)}

    def comprehensive_portfolio_analysis_without_agent_intelligence(self, symbols: List[str] = None) -> str:
        """
        Run comprehensive AI-powered portfolio analysis
        Orchestrate analysis across multiple MCP servers
        """
        if not symbols:
            # Get current portfolio symbols from the portfolio server
            try:
                portfolio_data = self.call_mcp_tool(self.portfolio_server, "get_portfolio_overview")
                if portfolio_data.get('status') == 'success' and portfolio_data.get('positions'):
                    symbols = [pos['symbol'] for pos in portfolio_data['positions']]
                else:
                    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]  # Default portfolio
            except:
                symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]  # Default portfolio if error occurs
        
        analysis_data = {
            'portfolio_summary': {},
            'market_data': {},
            'technical_analysis': {},
            'portfolio_metrics': {},
            'market_overview': {},
            'timestamp': datetime.now().isoformat()
        }

        try:
            # STEP 1: Get current portfolio summary and positions
            portfolio_summary = self.call_mcp_tool(self.portfolio_server, "get_portfolio_overview")
            analysis_data['portfolio_summary'] = portfolio_summary
            
            portfolio_metrics = self.call_mcp_tool(self.portfolio_server, "get_portfolio_metrics")
            analysis_data['portfolio_metrics'] = portfolio_metrics
            
            asset_allocation = self.call_mcp_tool(self.portfolio_server, "get_asset_allocation")
            analysis_data['asset_allocation'] = asset_allocation
            
            # STEP 2: For each stock, get current market data and technical analysis
            for symbol in symbols:
                # Get current market data
                stock_data = self.call_mcp_tool(self.market_data_server, "get_stock_price", symbol=symbol, period="1mo")
                analysis_data['market_data'][symbol] = stock_data
                
                # Get technical analysis
                technical_data = self.call_mcp_tool(self.market_data_server, "get_technical_analysis", symbol=symbol)
                analysis_data['technical_analysis'][symbol] = technical_data
            
            # STEP 3: Get overall market overview
            market_overview = self.call_mcp_tool(self.market_data_server, "get_market_overview")
            analysis_data['market_overview'] = market_overview
            
        except Exception as e:
            self.logger.error(f"Error gathering analysis data: {str(e)}")
        
        # STEP 4: Generate AI-powered analysis
        prompt = f"""
        Perform a comprehensive AI-powered analysis for portfolio: {symbols}
        
        STEP 1: Get current portfolio summary and positions
        STEP 2: For each stock, get:
            - Current market data and prices
            - AI sentiment analysis from recent news
            - AI-powered price predictions
        STEP 3: Generate portfolio-level insights:
            - Overall risk assessment
            - Rebalancing recommendations
            - New position suggestions
            - Market outlook
        
        STEP 4: Create actionable recommendations with:
            - Specific buy/sell/hold decisions
            - Position sizing recommendations
            - Risk management strategies
            - Timeline for implementation
        
        Present results in a professional, structured format with clear sections.
        """
        
        return self.agent.run(prompt)
    
    def comprehensive_portfolio_analysis(self, symbols: List[str] = None) -> str:
        prompt = f"""
        You are a professional financial advisor with access to real-time market data and portfolio information.
        Conduct a comprehensive portfolio analysis using the available tools.
        
        AVAILABLE TOOLS:
        Portfolio Management Server (http://localhost:8002):
            - get_portfolio_overview: Get current positions, P&L, and allocation
            - get_portfolio_metrics: Get Sharpe ratio, beta, volatility, max drawdown
            - calculate_returns: Get returns for various timeframes
            - get_asset_allocation: Get sector and symbol allocation breakdown
            
            Market Data Server (http://localhost:8001):
            - get_stock_price: Get current market data for any symbol
            - get_technical_analysis: Get RSI, MACD, moving averages, signals
            - get_market_overview: Get major market indices and trends
            - get_portfolio_performance: Calculate weighted portfolio performance
        
        ANALYSIS REQUIREMENTS:
        1. Start with portfolio overview to understand current positions
        2. For each significant holding (>5% allocation), get detailed technical analysis
        3. Always include portfolio risk metrics (Sharpe, beta, volatility)
        4. Get market context from major indices
        5. Use your professional judgment to determine what additional analysis is needed
        
        ADAPTIVE INTELLIGENCE GUIDELINES:
        - If portfolio is concentrated (>30% in one position), emphasize risk analysis
        - If any position shows strong technical signals, provide deeper analysis
        - If market indices show high volatility, focus on defensive strategies
        - If portfolio metrics show high risk, prioritize risk management recommendations
        - Use your expertise to identify the most valuable insights for this specific portfolio
        
        DELIVERABLE FORMAT:
        ## Executive Summary
        - Key findings and overall assessment
        - Primary recommendations

        ## Portfolio Overview
        - Current positions and allocation
        - Performance metrics and risk assessment

        ## Individual Position Analysis
        - Analysis of significant holdings
        - Technical and fundamental insights

        ## Market Context
        - Current market environment
        - Implications for the portfolio

        ## Risk Assessment
        - Key risk factors identified
        - Risk mitigation strategies

        ## Actionable Recommendations
        - Specific buy/sell/hold decisions
        - Position sizing recommendations
        - Risk management actions
        - Timeline for implementation

        Call the appropriate tools based on your professional judgment and provide comprehensive analysis.
        """
    
        if symbols:
            prompt += f"\n\nSpecific symbols to analyze: {symbols}"
    
        try:
            return self.agent.run(prompt)
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}")
            # Fallback to manual orchestration if agent fails
            # return self._fallback_portfolio_analysis(symbols) TODO: Implement if needed
    
    def stock_recommendation(self, symbol: str) -> str:
        """
        Generate AI-powered stock recommendation
        Analyze stock using multiple data sources
        """
     
        prompt = f"""
        You are a senior equity research analyst. Provide a comprehensive stock recommendation for {symbol}.
        
        AVAILABLE TOOLS:
        - get_stock_price: Get current price, fundamentals, and company information
        - get_technical_analysis: Get technical indicators and trading signals
        - get_market_overview: Get market context and sector performance
        
        ANALYSIS REQUIREMENTS:
        1. Get comprehensive stock data including fundamentals
        2. Perform technical analysis with indicators and signals
        3. Consider market context and sector trends
        4. Use your professional judgment for additional analysis needed
        
        RECOMMENDATION FORMAT:
        ## RECOMMENDATION: [BUY/HOLD/SELL] - [CONFIDENCE LEVEL]
        
        ## Investment Thesis
        - Key reasons supporting the recommendation
        - Fundamental strengths/weaknesses
        - Competitive position assessment
        
        ## Technical Analysis Summary
        - Key indicators and signals
        - Support/resistance levels
        - Trend and momentum analysis
        
        ## Valuation Assessment
        - Current valuation metrics
        - Fair value estimate
        - Risk-reward profile
        
        ## Risks and Catalysts
        - Upside catalysts
        - Risk factors
        - What could change the thesis
        
        ## Position Guidance
        - Recommended position size
        - Entry/exit strategy
        - Stop-loss levels
        
        Use the available tools intelligently to gather the data you need for this analysis.
        """

        try:
            return self.agent.run(prompt)
        except Exception as e:
            self.logger.error(f"Error in stock recommendation for {symbol}: {e}")
            return f"Error generating recommendation for {symbol}: {str(e)}"
    
    def portfolio_optimization_advice(self) -> str:
        """
        Provide portfolio optimization recommendations
        Use risk analysis and predictions for optimization advice
        """
        
            # Gather portfolio data
            # portfolio_summary = self.call_mcp_tool(self.portfolio_server, "get_portfolio_overview")
            # portfolio_metrics = self.call_mcp_tool(self.portfolio_server, "get_portfolio_metrics")
            # asset_allocation = self.call_mcp_tool(self.portfolio_server, "get_asset_allocation")
            # market_overview = self.call_mcp_tool(self.market_data_server, "get_market_overview")
            
            # # Get returns for multiple timeframes
            # returns_1m = self.call_mcp_tool(self.portfolio_server, "calculate_returns", timeframe="1M")
            # returns_3m = self.call_mcp_tool(self.portfolio_server, "calculate_returns", timeframe="3M")
            # returns_6m = self.call_mcp_tool(self.portfolio_server, "calculate_returns", timeframe="6M")
            
        prompt = f"""
        You are a portfolio optimization specialist. Analyze the current portfolio and provide optimization recommendations.
        
        AVAILABLE TOOLS:
        - get_portfolio_overview: Current positions and performance
        - get_portfolio_metrics: Risk metrics (Sharpe, beta, volatility)
        - get_asset_allocation: Sector and position allocation
        - calculate_returns: Multi-timeframe return analysis
        - get_market_overview: Market context for optimization
        
        ANALYSIS FRAMEWORK:
        1. Assess current portfolio composition and performance
        2. Analyze risk-return characteristics
        3. Evaluate diversification effectiveness
        4. Consider market environment for optimization timing
        5. Use professional judgment for additional analysis needed
        
        OPTIMIZATION DELIVERABLES:
        ## Portfolio Health Score (1-10)
        - Overall assessment and key issues
        
        ## Risk-Return Optimization
        - Current vs optimal risk-return profile
        - Sharpe ratio improvement opportunities
        
        ## Diversification Analysis
        - Concentration risks and solutions
        - Sector/geographic diversification gaps
        
        ## Rebalancing Strategy
        - Specific position adjustments
        - New positions to consider
        - Exit recommendations
        
        ## Implementation Plan
        - Step-by-step optimization roadmap
        - Priority order and timing
        - Risk management during transition
        
        Call the appropriate tools to gather the data you need for comprehensive optimization analysis.
        """
        
        try:
            return self.agent.run(prompt)
        except Exception as e:
            self.logger.error(f"Error in optimization advice: {e}")
            return f"Error generating optimization advice: {str(e)}"
    
    def market_outlook_analysis(self) -> str:
        """
        Generate market outlook based on multiple factors
        Analyze market sentiment, news, and technical indicators
        """
        # try:
        #     # Get comprehensive market data
        #     market_overview = self.call_mcp_tool(self.market_data_server, "get_market_overview")
            
        #     # Get technical analysis for major indices
        #     sp500_technical = self.call_mcp_tool(self.market_data_server, "get_technical_analysis", symbol="^GSPC")
        #     nasdaq_technical = self.call_mcp_tool(self.market_data_server, "get_technical_analysis", symbol="^IXIC")
            
        #     # Get data for key economic indicators via major ETFs
        #     vix_data = self.call_mcp_tool(self.market_data_server, "get_stock_price", symbol="^VIX", period="3mo")
        #     dxy_data = self.call_mcp_tool(self.market_data_server, "get_stock_price", symbol="DX-Y.NYB", period="3mo")
            
        #     prompt = f"""
        #     You are a chief market strategist. Provide a comprehensive market outlook analysis.

        #     MARKET OVERVIEW:
        #     {json.dumps(market_overview, indent=2)}

        #     S&P 500 TECHNICAL ANALYSIS:
        #     {json.dumps(sp500_technical, indent=2)}

        #     NASDAQ TECHNICAL ANALYSIS:
        #     {json.dumps(nasdaq_technical, indent=2)}

        #     FEAR & GREED INDICATOR (VIX):
        #     {json.dumps(vix_data, indent=2)}

        #     Provide a comprehensive market outlook:

        #     ## EXECUTIVE MARKET SUMMARY
        #     - Overall market sentiment assessment
        #     - Key market themes and trends
        #     - Market regime identification (bull/bear/transitional)

        #     ## TECHNICAL MARKET ANALYSIS
        #     - Major indices technical outlook
        #     - Key support and resistance levels
        #     - Trend analysis and momentum assessment
        #     - Volume and breadth analysis

        #     ## MARKET SENTIMENT INDICATORS
        #     - VIX and volatility analysis
        #     - Fear & greed assessment
        #     - Risk-on vs risk-off positioning

        #     ## SECTOR ROTATION ANALYSIS
        #     - Leading and lagging sectors
        #     - Rotation opportunities
        #     - Defensive vs growth positioning

        #     ## ECONOMIC BACKDROP
        #     - Interest rate environment impact
        #     - Inflation considerations
        #     - Economic cycle positioning

        #     ## RISK FACTORS AND CATALYSTS
        #     - Key upside catalysts for markets
        #     - Major risk factors to monitor
        #     - Black swan considerations

        #     ## INVESTMENT STRATEGY IMPLICATIONS
        #     - Recommended market positioning
        #     - Asset allocation guidance
        #     - Hedging considerations

        #     ## SHORT-TERM OUTLOOK (1-3 months)
        #     - Near-term market direction
        #     - Key levels to watch
        #     - Tactical positioning recommendations

        #     ## MEDIUM-TERM OUTLOOK (3-12 months)
        #     - Longer-term market trajectory
        #     - Structural trends to consider
        #     - Strategic positioning recommendations

        #     ## MONITORING CHECKLIST
        #     - Key indicators to track
        #     - Warning signals to watch
        #     - Review frequency recommendations

        #     Provide specific, actionable insights with clear reasoning and quantitative levels where applicable.
        #     """
            
        #     return self.agent.run(prompt)

        prompt = f"""
        You are a chief market strategist. Provide a comprehensive market outlook analysis.
        
        AVAILABLE TOOLS:
        - get_market_overview: Major indices and market performance
        - get_technical_analysis: Technical analysis of market indices (^GSPC, ^IXIC)
        - get_stock_price: Economic indicators via ETFs/indices
        
        ANALYSIS FRAMEWORK:
        1. Assess current market trends and momentum
        2. Analyze technical indicators for major indices
        3. Consider volatility and sentiment indicators
        4. Evaluate economic backdrop through market data
        5. Use professional judgment for comprehensive outlook
        
        MARKET OUTLOOK DELIVERABLES:
        ## Executive Market Summary
        - Current market regime assessment
        - Key themes and trends
        
        ## Technical Market Analysis
        - Major indices outlook
        - Key levels and trends
        - Volume and momentum analysis
        
        ## Risk Environment
        - Volatility assessment
        - Risk factors and catalysts
        - Defensive vs growth positioning
        
        ## Investment Implications
        - Strategic positioning recommendations
        - Sector and style preferences
        - Risk management considerations
        
        ## Outlook Timeline
        - Near-term (1-3 months) expectations
        - Medium-term (3-12 months) trajectory
        - Key events and catalysts to monitor
        
        Use the available tools intelligently to gather market data for comprehensive analysis.
        """
            
        try:
            return self.agent.run(prompt)
        except Exception as e:
            self.logger.error(f"Error in market outlook: {e}")
            return f"Error generating market outlook: {str(e)}"

class NewsAnalysisAgent:
    """Specialized agent for news and sentiment analysis"""
    
    # def __init__(self, model: str = "claude-3-sonnet-20240229"):
    #     self.logger = logging.getLogger(__name__)
    #     self.agent = Agent(model=model)
    #     self.news_sources = [
    #         "https://finance.yahoo.com/",
    #         "https://www.cnbc.com/business/",
    #         "https://www.bloomberg.com/markets",
    #         "https://www.reuters.com/business/"
    #     ]

    def __init__(self, model: str = "claude-3-sonnet-20240229"):
        self.logger = logging.getLogger(__name__)
        try:
            self.agent = Agent(model=model)
        except Exception as e:
            self.logger.warning(f"Could not initialize news agent: {e}")
            # Fallback model
            self.agent = Agent(model="gpt-3.5-turbo")
    
    def analyze_market_news(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Analyze news impact on portfolio symbols
        """
        try:
            news_analysis = {
                'overall_sentiment': {},
                'symbol_specific': {},
                'market_themes': [],
                'risk_alerts': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Overall market sentiment analysis
            overall_prompt = f"""
            You are a financial news analyst. Based on current market conditions and recent developments,
            provide an overall market sentiment analysis for the current environment.
            
            Consider recent trends in:
            - Economic indicators and Federal Reserve policy
            - Geopolitical developments
            - Sector rotation and market leadership
            - Volatility and risk sentiment
            - Corporate earnings trends
            
            Provide:
            1. OVERALL MARKET SENTIMENT SCORE (-10 to +10)
            2. KEY MARKET THEMES (top 3-5 driving factors)
            3. RISK FACTORS (major concerns)
            4. OPPORTUNITIES (areas of potential strength)
            5. TIMELINE OUTLOOK (short vs medium-term view)
            
            Format as structured analysis with clear reasoning.
            """
            
            overall_analysis = self.agent.run(overall_prompt)
            news_analysis['overall_sentiment'] = {
                'analysis': overall_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            # Symbol-specific analysis
            for symbol in symbols[:5]:  # Limit to 5 symbols to avoid rate limits
                symbol_prompt = f"""
                Analyze the current news sentiment and developments for {symbol}.
                
                Consider:
                - Recent company announcements and earnings
                - Sector-specific developments
                - Regulatory or competitive changes
                - Analyst recommendations and price targets
                - Market positioning and institutional flow
                
                Provide:
                1. SENTIMENT SCORE (-10 to +10) for {symbol}
                2. KEY DEVELOPMENTS affecting the stock
                3. RISK ALERTS for potential negative impacts
                4. OPPORTUNITY HIGHLIGHTS for positive catalysts
                5. PRICE IMPACT ASSESSMENT (likely direction and magnitude)
                
                Keep analysis focused and actionable.
                """
                
                try:
                    analysis = self.agent.run(symbol_prompt)
                    news_analysis['symbol_specific'][symbol] = {
                        'analysis': analysis,
                        'timestamp': datetime.now().isoformat()
                    }
                except Exception as e:
                    self.logger.error(f"Error analyzing news for {symbol}: {e}")
                    news_analysis['symbol_specific'][symbol] = {
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
            
            return news_analysis
            
        except Exception as e:
            self.logger.error(f"Error in news analysis: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
class RiskManagementAgent:
    """Specialized agent for risk management"""
    
    def __init__(self, model: str = "gpt-4"):
        self.logger = logging.getLogger(__name__)
        try:
            self.agent = Agent(model=model)
        except Exception as e:
            self.logger.warning(f"Could not initialize risk agent: {e}")
            self.agent = Agent(model="gpt-3.5-turbo")
    
    def assess_portfolio_risk(self, portfolio_data: Dict) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk assessment
        """
        try:
            prompt = f"""
            You are a senior risk management specialist. Conduct a comprehensive risk assessment.
            
            PORTFOLIO DATA:
            {json.dumps(portfolio_data, indent=2)}
            
            Provide detailed risk assessment:
            
            ## OVERALL RISK SCORE (1-10 scale)
            - Risk level and classification
            
            ## QUANTITATIVE RISK ANALYSIS
            - Sharpe ratio and risk-adjusted returns assessment
            - Beta and market correlation analysis
            - Volatility and maximum drawdown evaluation
            - Value at Risk interpretation
            
            ## CONCENTRATION RISK
            - Position concentration assessment
            - Single stock and sector exposure risks
            - Diversification effectiveness
            
            ## LIQUIDITY AND MARKET RISKS
            - Portfolio liquidity assessment
            - Market environment risks
            - Economic sensitivity analysis
            
            ## RISK ALERTS AND WARNINGS
            - Immediate risk concerns
            - Position-specific warnings
            - Market environment risks
            
            ## RISK MITIGATION STRATEGIES
            - Specific risk reduction recommendations
            - Position sizing adjustments
            - Hedging considerations
            - Diversification improvements
            
            ## MONITORING FRAMEWORK
            - Key metrics to track
            - Alert thresholds
            - Review schedule
            
            Provide specific, actionable recommendations with clear priorities.
            """
            
            risk_assessment = self.agent.run(prompt)
            
            return {
                'status': 'success',
                'risk_assessment': risk_assessment,
                'timestamp': datetime.now().isoformat(),
                'portfolio_analyzed': bool(portfolio_data and portfolio_data.get('status') == 'success')
            }
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class FinancialAICoordinator:
    """Coordinates all AI agents for comprehensive financial analysis"""
    
    def __init__(self):
        self.financial_agent = FinancialAgent()
        self.news_agent = NewsAnalysisAgent()
        self.risk_agent = RiskManagementAgent()
        self.logger = logging.getLogger(__name__)
    
    def run_full_analysis(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive analysis using all agents"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'comprehensive',
            'symbols_analyzed': symbols or []
        }
        
        try:
            # Main portfolio analysis with intelligent tool usage
            self.logger.info("Running comprehensive portfolio analysis...")
            results['portfolio_analysis'] = self.financial_agent.comprehensive_portfolio_analysis(symbols)
            
            # News and sentiment analysis
            if symbols:
                self.logger.info("Running news analysis...")
                results['news_analysis'] = self.news_agent.analyze_market_news(symbols)
            
            # Market outlook
            self.logger.info("Running market outlook analysis...")
            results['market_outlook'] = self.financial_agent.market_outlook_analysis()
            
            # Risk assessment
            self.logger.info("Running risk assessment...")
            portfolio_data = self.financial_agent.call_mcp_tool(
                self.financial_agent.portfolio_server, 
                "get_portfolio_overview"
            )
            results['risk_assessment'] = self.risk_agent.assess_portfolio_risk(portfolio_data)
            
            # Portfolio optimization
            self.logger.info("Running portfolio optimization...")
            results['optimization_advice'] = self.financial_agent.portfolio_optimization_advice()
            
            results['status'] = 'success'
            self.logger.info("Comprehensive analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in full analysis: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
# Utility functions for testing and validation
def test_mcp_connectivity() -> Dict[str, bool]:
    """Test connectivity to MCP servers"""
    servers = {
        "Portfolio Server": "http://localhost:8002",
        "Market Data Server": "http://localhost:8001"
    }
    
    connectivity = {}
    for name, url in servers.items():
        try:
            response = requests.get(url, timeout=5)
            connectivity[name] = True
        except:
            connectivity[name] = False
    
    return connectivity

def validate_agent_setup() -> Dict[str, Any]:
    """Validate agent setup and configuration"""
    validation = {
        'timestamp': datetime.now().isoformat(),
        'mcp_servers': test_mcp_connectivity(),
        'agent_initialization': {}
    }
    
    try:
        # Test agent initialization
        agent = FinancialAgent()
        validation['agent_initialization']['FinancialAgent'] = True
        
        news_agent = NewsAnalysisAgent()
        validation['agent_initialization']['NewsAnalysisAgent'] = True
        
        risk_agent = RiskManagementAgent()
        validation['agent_initialization']['RiskManagementAgent'] = True
        
        coordinator = FinancialAICoordinator()
        validation['agent_initialization']['FinancialAICoordinator'] = True
        
    except Exception as e:
        validation['agent_initialization']['error'] = str(e)
    
    return validation

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== FINANCIAL AI AGENT SYSTEM ===")
    print("Initializing agents and testing connectivity...")
    
    # Validate setup
    validation = validate_agent_setup()
    print(f"System validation: {validation}")
    
    # Initialize the coordinator
    coordinator = FinancialAICoordinator()
    
    # Test with sample portfolio
    test_symbols = ['AAPL', 'GOOGL', 'MSFT']
    print(f"\nRunning comprehensive analysis for: {test_symbols}")
    
    # Run comprehensive analysis
    results = coordinator.run_full_analysis(test_symbols)
    
    # Display results
    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"Status: {results['status']}")
    print(f"Timestamp: {results['timestamp']}")
    
    if results['status'] == 'success':
        print(f"\n✅ Portfolio Analysis: {'Available' if 'portfolio_analysis' in results else 'Failed'}")
        print(f"✅ News Analysis: {'Available' if 'news_analysis' in results else 'Failed'}")
        print(f"✅ Market Outlook: {'Available' if 'market_outlook' in results else 'Failed'}")
        print(f"✅ Risk Assessment: {'Available' if 'risk_assessment' in results else 'Failed'}")
        print(f"✅ Optimization Advice: {'Available' if 'optimization_advice' in results else 'Failed'}")
        
        # Show sample output
        if 'portfolio_analysis' in results:
            print(f"\n=== PORTFOLIO ANALYSIS PREVIEW ===")
            analysis = results['portfolio_analysis']
            preview = analysis[:500] + "..." if len(analysis) > 500 else analysis
            print(preview)
            
    else:
        print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
