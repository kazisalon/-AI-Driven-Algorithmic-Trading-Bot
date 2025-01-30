import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import yfinance as yf
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class TradingEnvironment(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        super(TradingEnvironment, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [balance, holdings, 5 price points, RSI]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(8,), 
            dtype=np.float32
        )

    def calculate_rsi(self) -> float:
        """Calculate RSI indicator"""
        prices = self.df['Close'].values
        current_idx = self.current_step
        
        # Calculate RSI
        delta = np.diff(prices[max(0, current_idx-15):current_idx+1])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def get_observation(self) -> np.ndarray:
        """Get current observation state"""
        prices = self.df['Close'].values[max(0, self.current_step-4):self.current_step+1]
        prices = np.pad(prices, (5-len(prices), 0), mode='edge')
        
        rsi = self.calculate_rsi()
        
        obs = np.array([
            self.balance,
            self.holdings,
            *prices,
            rsi
        ], dtype=np.float32)
        
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one time step within the environment"""
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Execute trading action
        reward = 0
        if action == 1:  # Buy
            if self.balance >= current_price:
                self.holdings += 1
                self.balance -= current_price
        elif action == 2:  # Sell
            if self.holdings > 0:
                self.holdings -= 1
                self.balance += current_price
        
        # Calculate portfolio value and return
        portfolio_value = self.balance + (self.holdings * current_price)
        prev_price = self.df['Close'].iloc[self.current_step - 1]
        prev_value = self.balance + (self.holdings * prev_price)
        
        # Calculate reward based on portfolio return
        pct_return = (portfolio_value - prev_value) / prev_value
        reward = pct_return * 100
        
        return self.get_observation(), reward, done, {}

    def reset(self):
        """Reset the environment"""
        self.balance = self.initial_balance
        self.holdings = 0
        self.current_step = 0
        return self.get_observation()

class TradingBot:
    """Simple trading bot using reinforcement learning"""
    
    def __init__(self, symbol: str, start_date: str, end_date: str):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._fetch_data()
        self.model = None
        
    def _fetch_data(self) -> pd.DataFrame:
        """Fetch historical data using yfinance"""
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=self.start_date, end=self.end_date)
        return df
    
    def train(self, total_timesteps: int = 10000):
        """Train the RL model"""
        env = TradingEnvironment(self.data)
        env = DummyVecEnv([lambda: env])
        
        self.model = PPO("MlpPolicy", env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)
    
    def backtest(self) -> pd.DataFrame:
        """Backtest the trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before backtesting")
            
        env = TradingEnvironment(self.data)
        obs = env.reset()
        done = False
        
        results = []
        while not done:
            action, _states = self.model.predict(obs)
            obs, reward, done, info = env.step(action)
            
            results.append({
                'date': self.data.index[env.current_step],
                'action': ['Hold', 'Buy', 'Sell'][action],
                'price': self.data['Close'].iloc[env.current_step],
                'portfolio_value': env.balance + (env.holdings * self.data['Close'].iloc[env.current_step]),
                'holdings': env.holdings,
                'balance': env.balance
            })
            
        return pd.DataFrame(results)
    
    def plot_results(self, results: pd.DataFrame):
        """Plot backtest results"""
        plt.figure(figsize=(12, 6))
        plt.plot(results['date'], results['portfolio_value'], label='Portfolio Value')
        plt.title('Trading Bot Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    # Example usage
    bot = TradingBot(
        symbol='AAPL',
        start_date='2022-01-01',
        end_date='2023-12-31'
    )
    
    # Train the model
    print("Training the model...")
    bot.train(total_timesteps=10000)
    
    # Backtest the model
    print("Running backtest...")
    results = bot.backtest()
    
    # Print final results
    initial_value = 10000
    final_value = results['portfolio_value'].iloc[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    print(f"\nBacktest Results:")
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
    # Plot results
    bot.plot_results(results)

if __name__ == "__main__":
    main()