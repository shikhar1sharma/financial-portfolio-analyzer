"""
Main Financial AI Agent
Coordinates between MCP servers to provide comprehensive analysis
"""

import openai
import json
from typing import Dict, Any, List
import logging
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FinancialAgent:
    """Main AI agent for financial analysis and coordination"""
    
    def __init__(self, model: str = "gpt-4"):
        self.logger = logging.getLogger(__name__)
        
        # Setup OpenAI client
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.model = model
        self.client = openai.OpenAI()

        # MCP servers endpoints - AWS hosted
        self.portfolio_server = "http://ec2-3-90-112-2.compute-1.amazonaws.com:8002"
        self.market_data_server = "http://ec2-3-90-112-2.compute-1.amazonaws.com:8001"
    
    def call_mcp_tool(self, server_url: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a tool on the specified MCP server
        """
        try:
            response = requests.post(
                f"{server_url}/tools/{tool_name}",
                json=kwargs,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error calling MCP tool {tool_name} on {server_url}: {e}")
            return {"error": f"Network error: {str(e)}"}
        except Exception as e:
            self.logger.error(f"Error calling MCP tool {tool_name} on {server_url}: {e}")
            return {"error": str(e)}

    def _call_openai(self, prompt: str, system_message: str = None) -> str:
        """
        Call OpenAI API with the given prompt
        """
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            return f"Error generating analysis: {str(e)}"

    def comprehensive_portfolio_analysis(self, symbols: List[str] = None) -> str:
        """
        Run comprehensive AI-powered portfolio analysis
        """
        # Gather data from MCP servers
        portfolio_data = self.call_mcp_tool(self.portfolio_server, "get_portfolio_overview")
        market_overview = self.call_mcp_tool(self.market_data_server, "get_market_overview")
        
        # Get symbols from portfolio if not provided
        if not symbols and portfolio_data.get('status') == 'success':
            symbols = [pos['symbol'] for pos in portfolio_data.get('positions', [])]
        elif not symbols:
            symbols = ["AAPL", "GOOGL", "MSFT"]  # Default symbols
        
        # Get technical analysis for key positions
        technical_analyses = {}
        for symbol in symbols[:5]:  # Limit to 5 symbols
            tech_data = self.call_mcp_tool(self.market_data_server, "get_technical_analysis", symbol=symbol)
            technical_analyses[symbol] = tech_data
        
        # Create comprehensive prompt
        system_message = """
        You are a professional financial advisor and portfolio manager. Provide comprehensive, 
        actionable investment analysis based on the data provided. Structure your response with 
        clear sections and specific recommendations.
        """
        
        prompt = f"""
        Conduct a comprehensive portfolio analysis based on the following data:

        PORTFOLIO DATA:
        {json.dumps(portfolio_data, indent=2)}

        MARKET OVERVIEW:
        {json.dumps(market_overview, indent=2)}

        TECHNICAL ANALYSES:
        {json.dumps(technical_analyses, indent=2)}

        Please provide a comprehensive analysis with the following structure:

        ## Executive Summary
        - Overall portfolio health assessment
        - Key findings and primary recommendations
        - Risk level evaluation

        ## Portfolio Overview
        - Current positions and allocation analysis
        - Performance metrics interpretation
        - Diversification assessment

        ## Individual Position Analysis
        - Analysis of significant holdings (>5% allocation)
        - Technical indicators interpretation
        - Buy/hold/sell recommendations for each position

        ## Market Context
        - Current market environment assessment
        - How market conditions affect this portfolio
        - Sector and style positioning

        ## Risk Assessment
        - Key risk factors identified
        - Concentration risks
        - Market sensitivity analysis

        ## Actionable Recommendations
        - Specific position adjustments (with percentages)
        - New positions to consider
        - Risk management actions
        - Timeline for implementation (immediate, 1 month, 3 months)

        ## Monitoring Plan
        - Key metrics to track
        - Review schedule
        - Alert thresholds

        Provide specific, quantitative recommendations where possible. Include reasoning for all recommendations.
        """
        
        return self._call_openai(prompt, system_message)
    
    def stock_recommendation(self, symbol: str) -> str:
        """
        Generate AI-powered stock recommendation
        """
        # Gather stock data
        stock_data = self.call_mcp_tool(self.market_data_server, "get_stock_price", symbol=symbol, period="6mo")
        technical_analysis = self.call_mcp_tool(self.market_data_server, "get_technical_analysis", symbol=symbol)
        market_overview = self.call_mcp_tool(self.market_data_server, "get_market_overview")
        
        system_message = """
        You are a senior equity research analyst. Provide detailed, professional stock 
        recommendations with clear reasoning and specific price targets.
        """
        
        prompt = f"""
        Provide a comprehensive stock recommendation for {symbol} based on the following data:

        STOCK DATA:
        {json.dumps(stock_data, indent=2)}

        TECHNICAL ANALYSIS:
        {json.dumps(technical_analysis, indent=2)}

        MARKET CONTEXT:
        {json.dumps(market_overview, indent=2)}

        Please provide:

        ## RECOMMENDATION: [BUY/HOLD/SELL] - Confidence: [HIGH/MEDIUM/LOW]

        ## Investment Thesis
        - Key reasons supporting the recommendation
        - Company fundamentals assessment
        - Competitive position analysis
        - Valuation rationale

        ## Technical Analysis Summary
        - Key technical indicators interpretation
        - Support and resistance levels
        - Trend analysis and momentum assessment
        - Trading signals analysis

        ## Price Target and Valuation
        - 12-month price target with reasoning
        - Upside/downside potential
        - Key valuation metrics assessment
        - Fair value estimate

        ## Risk Factors and Catalysts
        - Key upside catalysts (timeline and probability)
        - Major risk factors and mitigation
        - What could change the investment thesis
        - Scenario analysis (bull/base/bear case)

        ## Position Management
        - Recommended position size (% of portfolio)
        - Entry strategy and timing
        - Stop-loss levels and exit criteria
        - Holding period recommendation

        Provide specific price levels and percentage recommendations where applicable.
        """
        
        return self._call_openai(prompt, system_message)
    
    def portfolio_optimization_advice(self) -> str:
        """
        Provide portfolio optimization recommendations
        """
        # Gather portfolio data
        portfolio_summary = self.call_mcp_tool(self.portfolio_server, "get_portfolio_overview")
        market_overview = self.call_mcp_tool(self.market_data_server, "get_market_overview")
        
        system_message = """
        You are a portfolio optimization specialist. Provide actionable recommendations 
        to improve risk-adjusted returns and portfolio efficiency.
        """
        
        prompt = f"""
        Analyze the portfolio and provide optimization recommendations:

        PORTFOLIO DATA:
        {json.dumps(portfolio_summary, indent=2)}

        MARKET CONTEXT:
        {json.dumps(market_overview, indent=2)}

        Provide optimization analysis with:

        ## Portfolio Health Score (1-10)
        - Overall assessment with reasoning
        - Key strengths and weaknesses

        ## Risk-Return Optimization
        - Current risk-return profile assessment
        - Opportunities for improvement
        - Optimal allocation suggestions

        ## Diversification Analysis
        - Current diversification effectiveness
        - Concentration risks and solutions
        - Geographic and sector gaps

        ## Rebalancing Strategy
        - Specific position adjustments (with percentages)
        - Sell recommendations with rationale
        - New positions to consider

        ## Implementation Plan
        - Priority order of changes (Phase 1, 2, 3)
        - Timeline for implementation
        - Risk management during transition
        - Cost considerations

        Provide specific percentage allocations and actionable steps.
        """
        
        return self._call_openai(prompt, system_message)
    
    def market_outlook_analysis(self) -> str:
        """
        Generate market outlook based on multiple factors
        """
        # Get market data
        market_overview = self.call_mcp_tool(self.market_data_server, "get_market_overview")
        sp500_technical = self.call_mcp_tool(self.market_data_server, "get_technical_analysis", symbol="^GSPC")
        nasdaq_technical = self.call_mcp_tool(self.market_data_server, "get_technical_analysis", symbol="^IXIC")
        
        system_message = """
        You are a chief market strategist. Provide comprehensive market outlook with 
        specific implications for investment strategy.
        """
        
        prompt = f"""
        Provide comprehensive market outlook analysis:

        MARKET OVERVIEW:
        {json.dumps(market_overview, indent=2)}

        S&P 500 TECHNICAL:
        {json.dumps(sp500_technical, indent=2)}

        NASDAQ TECHNICAL:
        {json.dumps(nasdaq_technical, indent=2)}

        Provide analysis with:

        ## Executive Market Summary
        - Current market regime (bull/bear/transitional)
        - Key themes driving markets
        - Overall sentiment assessment

        ## Technical Market Analysis
        - Major indices outlook and key levels
        - Trend analysis and momentum
        - Volume and breadth considerations

        ## Risk Environment Assessment
        - Current volatility environment
        - Key risk factors to monitor
        - Market stress indicators

        ## Sector and Style Outlook
        - Leading and lagging sectors
        - Growth vs value positioning
        - Defensive vs cyclical preferences

        ## Investment Strategy Implications
        - Recommended market positioning
        - Asset allocation guidance
        - Risk management considerations

        ## Outlook Timeline
        - Near-term (1-3 months) expectations with key levels
        - Medium-term (3-12 months) trajectory
        - Key events and catalysts to monitor

        Include specific index levels and percentage probability assessments where possible.
        """
        
        return self._call_openai(prompt, system_message)

class NewsAnalysisAgent:
    """Specialized agent for news and sentiment analysis"""
    
    def __init__(self, model: str = "gpt-4"):
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.client = openai.OpenAI()
    
    def _call_openai(self, prompt: str, system_message: str = None) -> str:
        """Call OpenAI API"""
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=3000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in news analysis: {str(e)}"
    
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
            
            # Overall market sentiment
            overall_prompt = """
            Based on current market conditions and recent developments, provide an overall 
            market sentiment analysis. Consider recent trends in economic indicators, 
            Federal Reserve policy, geopolitical developments, sector rotation, and 
            corporate earnings trends.

            Provide:
            1. OVERALL MARKET SENTIMENT SCORE (-10 to +10)
            2. KEY MARKET THEMES (top 3-5 driving factors)
            3. RISK FACTORS (major concerns)
            4. OPPORTUNITIES (areas of potential strength)
            5. TIMELINE OUTLOOK (short vs medium-term view)
            """
            
            system_message = "You are a financial news analyst providing market sentiment analysis."
            overall_analysis = self._call_openai(overall_prompt, system_message)
            
            news_analysis['overall_sentiment'] = {
                'analysis': overall_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            # Symbol-specific analysis
            for symbol in symbols[:5]:
                symbol_prompt = f"""
                Analyze current news sentiment for {symbol}. Consider recent company 
                announcements, earnings, sector developments, regulatory changes, 
                analyst recommendations, and market positioning.

                Provide:
                1. SENTIMENT SCORE (-10 to +10) for {symbol}
                2. KEY DEVELOPMENTS affecting the stock
                3. RISK ALERTS for potential negative impacts
                4. OPPORTUNITY HIGHLIGHTS for positive catalysts
                5. PRICE IMPACT ASSESSMENT (likely direction and magnitude)
                """
                
                try:
                    analysis = self._call_openai(symbol_prompt, system_message)
                    news_analysis['symbol_specific'][symbol] = {
                        'analysis': analysis,
                        'timestamp': datetime.now().isoformat()
                    }
                except Exception as e:
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
        self.model = model
        self.client = openai.OpenAI()
    
    def _call_openai(self, prompt: str, system_message: str = None) -> str:
        """Call OpenAI API"""
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for risk analysis
                max_tokens=3000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in risk analysis: {str(e)}"
    
    def assess_portfolio_risk(self, portfolio_data: Dict) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk assessment
        """
        try:
            system_message = """
            You are a senior risk management specialist. Provide comprehensive, quantitative 
            risk assessment with specific recommendations and actionable risk mitigation strategies.
            """
            
            prompt = f"""
            Conduct comprehensive portfolio risk assessment:

            PORTFOLIO DATA:
            {json.dumps(portfolio_data, indent=2)}

            Provide detailed risk assessment:

            ## OVERALL RISK SCORE (1-10 scale)
            - Risk level classification with reasoning
            - Risk-return profile assessment

            ## QUANTITATIVE RISK ANALYSIS
            - Portfolio concentration analysis
            - Volatility and correlation assessment
            - Liquidity risk evaluation
            - Market sensitivity analysis

            ## SPECIFIC RISK FACTORS
            - Position-specific risks
            - Sector concentration risks
            - Geographic exposure risks
            - Currency risks (if applicable)

            ## STRESS TESTING SCENARIOS
            - Market crash scenario (-20% market drop)
            - Sector rotation impact
            - Interest rate shock effects
            - Recession scenario analysis

            ## RISK MITIGATION STRATEGIES
            - Immediate risk reduction actions
            - Position sizing adjustments
            - Hedging recommendations
            - Diversification improvements

            ## MONITORING FRAMEWORK
            - Key risk metrics to track daily/weekly
            - Risk limit recommendations
            - Alert thresholds and triggers
            - Review schedule and rebalancing criteria

            ## ACTION PLAN
            - Priority 1: Immediate actions (within 1 week)
            - Priority 2: Medium-term actions (1-4 weeks)
            - Priority 3: Long-term risk management (1-3 months)

            Provide specific, quantitative recommendations with clear implementation steps.
            """
            
            risk_assessment = self._call_openai(prompt, system_message)
            
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
            # Main portfolio analysis
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
        "Portfolio Server": "http://ec2-3-90-112-2.compute-1.amazonaws.com:8002",
        "Market Data Server": "http://ec2-3-90-112-2.compute-1.amazonaws.com:8001"
    }
    
    connectivity = {}
    for name, url in servers.items():
        try:
            response = requests.get(url, timeout=10)
            connectivity[name] = True
        except:
            connectivity[name] = False
    
    return connectivity

def validate_agent_setup() -> Dict[str, Any]:
    """Validate agent setup and configuration"""
    validation = {
        'timestamp': datetime.now().isoformat(),
        'mcp_servers': test_mcp_connectivity(),
        'agent_initialization': {},
        'api_keys': {}
    }
    
    # Check API keys
    validation['api_keys']['openai'] = bool(os.getenv('OPENAI_API_KEY'))
    validation['api_keys']['anthropic'] = bool(os.getenv('ANTHROPIC_API_KEY'))
    
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
    print(f"\nSystem validation:")
    print(f"MCP Servers: {validation['mcp_servers']}")
    print(f"API Keys: {validation['api_keys']}")
    print(f"Agent Initialization: {validation['agent_initialization']}")
    
    # Quick connectivity test
    if all(validation['mcp_servers'].values()):
        print("\n✅ MCP servers are accessible")
        
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
    else:
        print("❌ MCP servers not accessible. Please check your AWS server status.")