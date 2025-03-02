# pairs_trading_daily_auto.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ib_insync import IB, Stock, util
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from itertools import combinations
import logging
import datetime
import time
import os

# Set Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
sns.set(style="whitegrid")

# IBKR Credentials
IB_HOST = '127.0.0.1'
IB_PORT = 7497  # Paper trading
CLIENT_ID = 123

# Strategy parameters
HEDGE_WINDOW = 60
Z_WINDOW = 60
ENTRY_ZSCORE = 1.5
EXIT_ZSCORE = 0.5
TRADE_QTY = 100  # Adjust based on your IB paper account balance

# Preferred daily ticker list
TICKER_LIST = ['AAPL', 'MSFT', 'ADBE', 'ORCL', 'IBM', 'AMD', 'INTC', 'EBAY', 'CSCO', 'QCOM']

def fetch_ib_data(tickers, duration='1 Y', bar_size='1 day'):
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    df_dict = {}
    for ticker in tickers:
        bars = ib.reqHistoricalData(Stock(ticker, 'SMART', 'USD'),
                                    endDateTime='',
                                    durationStr=duration,
                                    barSizeSetting=bar_size,
                                    whatToShow='ADJUSTED_LAST',
                                    useRTH=True,
                                    formatDate=1)
        df = util.df(bars).set_index('date')['close'].rename(ticker)
        df_dict[ticker] = df
        time.sleep(1)  # IB pacing rule
    ib.disconnect()
    return pd.concat(df_dict, axis=1).dropna()

def best_cointegrated_pair(df):
    pairs = combinations(df.columns,2)
    results = []
    for s1, s2 in pairs:
        _, pvalue, _ = coint(df[s1], df[s2])
        results.append((s1, s2, pvalue))
    return sorted(results, key=lambda x: x[2])[0]

def rolling_hedge_ratio(S1, S2, window=60):
    return sm.OLS(S2[-window:], sm.add_constant(S1[-window:])).fit().params[1]

def get_signals(S1, S2, hedge, z_window=60):
    spread = S2 - hedge * S1
    zscore = (spread - spread.rolling(z_window).mean())/spread.rolling(z_window).std()
    if zscore.iloc[-1] > ENTRY_ZSCORE:
        return -1  # Short spread: Short S2, Buy S1
    elif zscore.iloc[-1] < -ENTRY_ZSCORE:
        return 1  # Long spread: Buy S2, Short S1
    elif abs(zscore.iloc[-1]) < EXIT_ZSCORE:
        return 0  # Exit all positions
    return None

def execute_trade(pair, action):
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    stk1 = Stock(pair[0], 'SMART','USD')
    stk2 = Stock(pair[1], 'SMART','USD')
    ib.qualifyContracts(stk1, stk2)
    if action == 1:  # Long pair: Buy stk2, Sell stk1
        order1 = ib.marketOrder('SELL', TRADE_QTY)
        order2 = ib.marketOrder('BUY', TRADE_QTY)
    elif action == -1:  # Short pair: Sell stk2, Buy stk1
        order1 = ib.marketOrder('BUY', TRADE_QTY)
        order2 = ib.marketOrder('SELL', TRADE_QTY)
    elif action == 0:  # Exit
        logging.info("Closing all existing positions in pair.")
        positions = ib.positions()
        for pos in positions:
            if pos.contract.symbol in pair:
                action = 'SELL' if pos.position > 0 else 'BUY'
                order = ib.marketOrder(action, abs(pos.position))
                ib.placeOrder(pos.contract, order)
    else:
        logging.info("No clear signal today, skipping trades.")
        ib.disconnect()
        return
    if action != 0:
        ib.placeOrder(stk1, order1)
        ib.placeOrder(stk2, order2)
        logging.info(f"Executed Trades: {pair[0]} {order1.action}, {pair[1]} {order2.action}")
    ib.disconnect()

def plot_daily_chart(S1, S2, pair, hedge, z_window):
    spread = S2 - hedge * S1
    zscore = (spread - spread.rolling(z_window).mean())/spread.rolling(z_window).std()
    fig, axs = plt.subplots(3,1, figsize=(12,14), sharex=True)
    
    axs[0].plot(S1,label=pair[0], color='b')
    axs[0].plot(S2,label=pair[1], color='orange')
    axs[0].legend()
    axs[0].set_title("Asset Prices")
    
    axs[1].plot(spread,label='Spread',color='purple')
    axs[1].axhline(spread.mean(), color='red')
    axs[1].set_title("Spread between assets")
    axs[1].legend()
    
    axs[2].plot(zscore,label='Spread Z-score',color='green')
    axs[2].axhline(ENTRY_ZSCORE,color='red',linestyle='--')
    axs[2].axhline(-ENTRY_ZSCORE,color='red',linestyle='--')
    axs[2].axhline(EXIT_ZSCORE,color='blue',linestyle='--')
    axs[2].axhline(-EXIT_ZSCORE,color='blue',linestyle='--')
    axs[2].set_title("Z-Score of Spread")
    axs[2].legend()

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    plt.savefig(f"{today}_{pair[0]}_{pair[1]}_daily_strategy.png")
    plt.show()

def daily_run():
    logging.info("🚀 Starting Automated Daily Pairs-Trading...")
    df = fetch_ib_data(TICKER_LIST)
    pair = best_cointegrated_pair(df)
    logging.info(f"✅ Today's selected pair: {pair[0]} & {pair[1]}")
    
    S1, S2 = df[pair[0]], df[pair[1]]
    hedge_ratio = rolling_hedge_ratio(S1, S2, HEDGE_WINDOW)
    logging.info(f"Hedge Ratio: {hedge_ratio:.4f}")

    signal = get_signals(S1, S2, hedge_ratio, Z_WINDOW)
    logging.info(f"Today's Signal: {signal}")

    execute_trade(pair, signal)
    plot_daily_chart(S1, S2, pair, hedge_ratio, Z_WINDOW)

if __name__=="__main__":
    daily_run()