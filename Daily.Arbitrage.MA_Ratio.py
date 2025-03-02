# pairs_trading_strategy.py
# Optimized pairs trading strategy with IBKR API.
# Author: (Your Name or your preferred alias)
# Date: (todays date)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ib_insync import IB, Stock, util
import logging
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# IBKR settings
HOST = '127.0.0.1'
PORT = 7497  # Paper trading TWS port, real account uses 7496
CLIENT_ID = 1

# Parameters (can optimize later)
short_window = 5
long_window = 60
entry_zscore = 1.0
exit_zscore = 0.5

# Connect to IB 
def fetch_historical_data(ticker, duration='2 Y', barsize='1 day'):
    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    contract = Stock(ticker, 'SMART', 'USD')
    bars = ib.reqHistoricalData(
        contract, endDateTime='', durationStr=duration,
        barSizeSetting=barsize, whatToShow='ADJUSTED_LAST', useRTH=True
    )
    ib.disconnect()
    df = util.df(bars).set_index('date')
    return df['close']

# Fetch pair data
def fetch_pair(ticker1, ticker2):
    S1 = fetch_historical_data(ticker1)
    S2 = fetch_historical_data(ticker2)
    df = pd.concat([S1, S2], axis=1).dropna()
    df.columns = ['S1', 'S2']
    return df

# Calculate z-score & signals
def generate_signals(df):
    df['ratio'] = df['S1'] / df['S2']
    df['short_ma'] = df['ratio'].rolling(window=short_window).mean()
    df['long_ma'] = df['ratio'].rolling(window=long_window).mean()
    df['long_std'] = df['ratio'].rolling(window=long_window).std()
    df.dropna(inplace=True)
    df['zscore'] = (df['short_ma'] - df['long_ma']) / df['long_std']
    
    # generate signals
    df['positions'] = np.where(df['zscore'] > entry_zscore, -1, np.nan)
    df['positions'] = np.where(df['zscore'] < -entry_zscore, 1, df['positions'])
    df['positions'] = np.where(abs(df['zscore']) < exit_zscore, 0, df['positions'])
    df['positions'].ffill(inplace=True)
    df['positions'].fillna(0, inplace=True)
    
    return df

# Quick backtester
def backtest_signals(df):
    df['pair_ret'] = df['positions'].shift(1) * (np.log(df['S1']/df['S1'].shift(1)) - np.log(df['S2']/df['S2'].shift(1)))
    df['cum_ret'] = df['pair_ret'].cumsum()
    plt.figure(figsize=(12,6))
    plt.plot(df['cum_ret'])
    plt.title('Cumulative Return of Pairs Strategy')
    plt.ylabel('Cumulative Returns (log)')
    plt.xlabel('Date')
    plt.grid()
    plt.show()

# Place orders (demo ONLY), size=100 shares per leg (adjust as needed).
def place_ib_orders(ticker1, ticker2, position):
    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    contract1 = Stock(ticker1, 'SMART', 'USD')
    contract2 = Stock(ticker2, 'SMART', 'USD')
    ib.qualifyContracts(contract1, contract2)

    qty = 100  # demo size, adjust per your needs
    if position == 1:  # long S1, short S2 ratio
        order1 = ib.marketOrder('BUY', qty)
        order2 = ib.marketOrder('SELL', qty)
    elif position == -1:  # short S1, long S2 ratio
        order1 = ib.marketOrder('SELL', qty)
        order2 = ib.marketOrder('BUY', qty)
    else:
        logging.info("Position neutral, no orders placed.")
        ib.disconnect()
        return

    ib.placeOrder(contract1, order1)
    ib.placeOrder(contract2, order2)
    logging.info(f"Orders placed: {ticker1} {order1.action} and {ticker2} {order2.action}")
    ib.disconnect()

# Main routine to run daily
def run_daily_strategy(ticker1, ticker2):
    today = datetime.datetime.today().weekday()
    if today >= 5:
        logging.info("Weekend detected; markets closed.")
        return

    df = fetch_pair(ticker1, ticker2)
    df_signals = generate_signals(df)

    # Plot latest signals.
    logging.info(f"Latest Z-score: {df_signals['zscore'][-1]:.2f}. Current Position: {df_signals['positions'][-1]}")
    
    # Place today's trades based on yesterday's closing position
    latest_signal = df_signals['positions'].iloc[-1]
    place_ib_orders(ticker1, ticker2, latest_signal)

    # Optional: visualize
    backtest_signals(df_signals)

# Main execution entry
if __name__ == '__main__':
    print("👋 Welcome to IB pairs trading daily execution!")
    ticker1 = input("Please enter the first ticker (example: ADBE): ").strip().upper()
    ticker2 = input("Please enter the second ticker (example: MSFT): ").strip().upper()
    run_daily_strategy(ticker1, ticker2)