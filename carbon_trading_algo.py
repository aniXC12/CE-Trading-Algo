import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

@dataclass
class Company:
    """Represents an energy company with carbon credit exposure"""
    ticker: str
    name: str
    sector: str  # 'oil_gas', 'renewable', 'utility', 'industrial'
    carbon_credits_held: float  # in metric tons CO2e
    carbon_intensity: float  # tons CO2 per unit revenue
    esg_score: float  # 0-100
    market_cap: float
    
@dataclass
class Trade:
    """Represents a trade decision"""
    timestamp: datetime
    ticker: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: int
    price: float
    confidence: float
    reason: str

class CarbonCreditTradingAlgo:
    """
    Trading algorithm focused on energy companies with carbon credit exposure.
    Implements multiple strategies based on carbon market dynamics.
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.capital = initial_capital
        self.portfolio: Dict[str, int] = {}
        self.trade_history: List[Trade] = []
        self.carbon_price_history: List[float] = []
        
        # Algorithm parameters
        self.params = {
            'carbon_beta_threshold': 0.7,  # Correlation to carbon prices
            'esg_min_score': 60,
            'volatility_window': 20,
            'momentum_window': 10,
            'position_size_pct': 0.15,  # Max 15% per position
            'stop_loss_pct': 0.08,
            'take_profit_pct': 0.20
        }
        
        # Sample companies for demonstration
        self.companies = self._initialize_companies()
        
    def _initialize_companies(self) -> List[Company]:
        """Initialize sample energy companies with carbon exposure"""
        return [
            Company('NEE', 'NextEra Energy', 'renewable', 5000000, 0.15, 82, 150e9),
            Company('XOM', 'Exxon Mobil', 'oil_gas', 2000000, 2.5, 45, 400e9),
            Company('CVX', 'Chevron', 'oil_gas', 1800000, 2.3, 48, 280e9),
            Company('DUK', 'Duke Energy', 'utility', 3000000, 0.8, 68, 75e9),
            Company('ENPH', 'Enphase Energy', 'renewable', 50000, 0.05, 75, 20e9),
            Company('OXY', 'Occidental Petroleum', 'oil_gas', 4000000, 1.9, 52, 45e9),
            Company('AEP', 'American Electric Power', 'utility', 2500000, 0.9, 65, 50e9),
            Company('FSLR', 'First Solar', 'renewable', 30000, 0.03, 78, 25e9),
        ]
    
    def calculate_carbon_beta(self, company: Company, 
                             carbon_prices: np.ndarray, 
                             stock_returns: np.ndarray) -> float:
        """Calculate beta coefficient relative to carbon credit prices"""
        if len(carbon_prices) < 2 or len(stock_returns) < 2:
            return 0.0
        
        carbon_returns = np.diff(carbon_prices) / carbon_prices[:-1]
        
        # Ensure equal length
        min_len = min(len(carbon_returns), len(stock_returns))
        carbon_returns = carbon_returns[-min_len:]
        stock_returns = stock_returns[-min_len:]
        
        covariance = np.cov(stock_returns, carbon_returns)[0, 1]
        variance = np.var(carbon_returns)
        
        return covariance / variance if variance != 0 else 0.0
    
    def calculate_carbon_credit_score(self, company: Company, 
                                     carbon_price: float) -> float:
        """
        Calculate composite score based on carbon credit holdings and dynamics.
        Higher score = more attractive for trading.
        """
        # Normalize metrics
        credit_value = company.carbon_credits_held * carbon_price
        credit_to_mcap = credit_value / company.market_cap
        
        # Score components
        credit_score = min(credit_to_mcap * 100, 100)  # Cap at 100
        intensity_score = max(0, 100 - (company.carbon_intensity * 20))
        esg_score = company.esg_score
        
        # Weighted composite
        weights = {'credit': 0.4, 'intensity': 0.3, 'esg': 0.3}
        composite = (
            credit_score * weights['credit'] +
            intensity_score * weights['intensity'] +
            esg_score * weights['esg']
        )
        
        return composite
    
    def generate_signals(self, company: Company, 
                        market_data: Dict,
                        carbon_price: float) -> Dict:
        """Generate trading signals based on multiple factors"""
        
        signals = {
            'carbon_score': self.calculate_carbon_credit_score(company, carbon_price),
            'technical': 0,
            'fundamental': 0,
            'sentiment': 0
        }
        
        # Technical signal (simplified momentum)
        if 'price_history' in market_data:
            prices = np.array(market_data['price_history'])
            if len(prices) >= self.params['momentum_window']:
                momentum = (prices[-1] / prices[-self.params['momentum_window']] - 1) * 100
                signals['technical'] = np.clip(momentum * 5, -100, 100)
        
        # Fundamental signal based on carbon exposure
        if company.sector == 'renewable':
            signals['fundamental'] = 70  # Bullish on renewables
        elif company.sector == 'oil_gas' and company.carbon_credits_held > 2000000:
            signals['fundamental'] = 40  # Neutral on oil/gas with high credits
        else:
            signals['fundamental'] = 20
        
        # Sentiment based on ESG trend
        signals['sentiment'] = company.esg_score - 50  # Baseline at 50
        
        return signals
    
    def make_trade_decision(self, company: Company, 
                           signals: Dict,
                           current_price: float) -> Trade:
        """Make trading decision based on signals"""
        
        # Weighted signal aggregation
        weights = {'carbon_score': 0.35, 'technical': 0.25, 
                   'fundamental': 0.25, 'sentiment': 0.15}
        
        composite_signal = sum(signals[k] * weights[k] for k in weights)
        
        # Normalize to -100 to 100
        composite_signal = np.clip(composite_signal, -100, 100)
        
        # Decision logic
        action = 'HOLD'
        confidence = abs(composite_signal) / 100
        quantity = 0
        reason = f"Composite signal: {composite_signal:.1f}"
        
        if composite_signal > 60:
            action = 'BUY'
            max_position_value = self.capital * self.params['position_size_pct']
            quantity = int(max_position_value / current_price)
            reason += " | Strong bullish signals on carbon dynamics"
            
        elif composite_signal < -40:
            action = 'SELL'
            if company.ticker in self.portfolio:
                quantity = self.portfolio[company.ticker]
            reason += " | Bearish signals, reducing exposure"
        
        trade = Trade(
            timestamp=datetime.now(),
            ticker=company.ticker,
            action=action,
            quantity=quantity,
            price=current_price,
            confidence=confidence,
            reason=reason
        )
        
        return trade
    
    def execute_trade(self, trade: Trade):
        """Execute trade and update portfolio"""
        if trade.action == 'BUY' and trade.quantity > 0:
            cost = trade.quantity * trade.price
            if cost <= self.capital:
                self.capital -= cost
                self.portfolio[trade.ticker] = self.portfolio.get(trade.ticker, 0) + trade.quantity
                self.trade_history.append(trade)
                print(f"✓ BUY {trade.quantity} shares of {trade.ticker} @ ${trade.price:.2f}")
                
        elif trade.action == 'SELL' and trade.quantity > 0:
            if trade.ticker in self.portfolio and self.portfolio[trade.ticker] >= trade.quantity:
                proceeds = trade.quantity * trade.price
                self.capital += proceeds
                self.portfolio[trade.ticker] -= trade.quantity
                if self.portfolio[trade.ticker] == 0:
                    del self.portfolio[trade.ticker]
                self.trade_history.append(trade)
                print(f"✓ SELL {trade.quantity} shares of {trade.ticker} @ ${trade.price:.2f}")
    
    def run_strategy(self, market_data_feed: Dict, carbon_price: float):
        """Main strategy execution loop"""
        print(f"\n{'='*60}")
        print(f"Running Carbon Credit Trading Strategy")
        print(f"Carbon Credit Price: ${carbon_price:.2f}/ton CO2e")
        print(f"{'='*60}\n")
        
        for company in self.companies:
            # Get market data for this company
            ticker_data = market_data_feed.get(company.ticker, {})
            current_price = ticker_data.get('current_price', 100)
            
            # Generate signals
            signals = self.generate_signals(company, ticker_data, carbon_price)
            
            # Make decision
            trade = self.make_trade_decision(company, signals, current_price)
            
            # Execute if actionable
            if trade.action != 'HOLD':
                print(f"\n{company.name} ({company.ticker})")
                print(f"  Carbon Credits: {company.carbon_credits_held:,.0f} tons")
                print(f"  ESG Score: {company.esg_score}")
                print(f"  Signal: {trade.reason}")
                self.execute_trade(trade)
        
        self.print_portfolio_summary()
    
    def print_portfolio_summary(self):
        """Print current portfolio status"""
        print(f"\n{'='*60}")
        print(f"Portfolio Summary")
        print(f"{'='*60}")
        print(f"Cash: ${self.capital:,.2f}")
        print(f"Positions: {len(self.portfolio)}")
        for ticker, shares in self.portfolio.items():
            print(f"  {ticker}: {shares} shares")
        print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Initialize algorithm
    algo = CarbonCreditTradingAlgo(initial_capital=500000)
    
    # Simulate market data
    market_data = {
        'NEE': {'current_price': 65.50, 'price_history': np.random.randn(30).cumsum() + 65},
        'XOM': {'current_price': 108.20, 'price_history': np.random.randn(30).cumsum() + 108},
        'CVX': {'current_price': 145.80, 'price_history': np.random.randn(30).cumsum() + 145},
        'DUK': {'current_price': 95.30, 'price_history': np.random.randn(30).cumsum() + 95},
        'ENPH': {'current_price': 125.60, 'price_history': np.random.randn(30).cumsum() + 125},
        'OXY': {'current_price': 58.90, 'price_history': np.random.randn(30).cumsum() + 58},
        'AEP': {'current_price': 88.40, 'price_history': np.random.randn(30).cumsum() + 88},
        'FSLR': {'current_price': 210.30, 'price_history': np.random.randn(30).cumsum() + 210},
    }
    
    # Current carbon credit price (example: $85 per ton CO2e)
    carbon_credit_price = 85.0
    
    # Run the strategy
    algo.run_strategy(market_data, carbon_credit_price)
    
    print("\nAlgorithm Parameters:")
    for key, value in algo.params.items():
        print(f"  {key}: {value}")
