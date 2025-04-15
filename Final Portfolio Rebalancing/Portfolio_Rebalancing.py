#%%
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.optimize as sco
import json
#%%
tickers = ['AMZN', 'BA', 'CAT', 'GOOGL', 'GS', 'NKE', 'NVDA', 'SOFI', 'TSLA', 'UNH']
#%%
# Parameters
train_start = "2023-11-17"
test_start = "2024-03-01"
test_end = "2025-01-16"
initial_capital = 50000
lookback_days = 60

# Download both Close and Open prices to obtain the desired weights for each month
data = yf.download(tickers, start=train_start, end=test_end, progress=False)
close = data['Close'].dropna()
openp = data['Open'].dropna()

# Log returns (from close prices)
log_returns = np.log(close / close.shift(1)).dropna()

# Generate monthly rebalance dates starting from test_start
rebalance_dates = pd.date_range(start=test_start, end=test_end, freq='MS')  # Month Start

# Map to actual trading days (forward-fill if it's not a trading day)
rebalance_dates = [close.index[close.index.get_indexer([d], method='bfill')[0]] for d in rebalance_dates]

# Init
weights_per_month = {}
shares_held = pd.Series(0, index=tickers)

for date in rebalance_dates:
    # Ensure date exists in index
    if date not in close.index:
        date = close.index[close.index.get_indexer([date], method='bfill')[0]]
    
    end_idx = close.index.get_loc(date)
    start_idx = end_idx - lookback_days
    if start_idx < 0:
        continue

    # Get past data window (only up to yesterday)
    window_returns = log_returns.iloc[start_idx:end_idx]
    mean_returns = window_returns.mean() * 252
    cov_matrix = window_returns.cov() * 252

    # Define Return optimizer
    def neg_return(weights):
        port_return = np.dot(weights, mean_returns.values)
        return -port_return

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0.02, 0.7)] * len(tickers)
    bounds[-4] = (0.2, 0.7) # put more weights on NVDA
    bounds[-2] = (0.4, 0.8) # put more weights on TSLA
    init_guess = np.array([1 / len(tickers)] * len(tickers))
    result = sco.minimize(neg_return, init_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    if not result.success:
        continue

    weights = pd.Series(result.x, index=tickers)

    # Log weights
    weights_per_month[pd.to_datetime(date).strftime("%Y-%m-%d")] = weights.round(4).to_dict()
    
# Weights per Month
print(" Diversified Weights per Month:")
print(json.dumps(weights_per_month, indent=2))

#%%
predictions = {}
for stock in tickers:
    predictions[stock] = pd.read_csv('Predictions of ' + stock + '.csv', parse_dates=True, index_col=0)
    
def calculate_portfolio_daily_report():
    """
    Computes a daily portfolio value report for the test period (from 2024-03-01 to 2025-01-16)
    using the following methodology:
    
    1. On the first test day (or the next available trading day if 2024-03-01 isn't in the data),
       allocate the current portfolio value according to the weights provided for that period.
    2. For each day in the period until the next rebalancing date, update each stock's 
       allocated capital using the ratio of its predicted cumulative return on that day to the
       cumulative return at the period's start.
    3. On each day, the total portfolio value is computed by summing all per-stock values plus any
       uninvested capital.
    4. At each rebalancing day (adjusted to an actual trading day), the portfolio value is updated,
       and new allocations are made for the next period.
    
    This function returns a DataFrame indexed by day with detailed portfolio values.
    """
    portfolio_value = initial_capital
    rebalance_dates_sorted = list(weights_per_month.keys())
    daily_records = []
    
    # Index of trading dates from 1 March 2024 until 16 January 2025
    pred_index = pd.to_datetime(predictions[tickers[0]].index)
    
    for i in range(len(rebalance_dates_sorted)):
        # Convert the desired rebalance date to datetime and adjust to the next trading day if needed.
        desired_start = pd.to_datetime(rebalance_dates_sorted[i])
        period_start = pred_index[pred_index.get_indexer([desired_start], method='bfill')[0]]
        
        # Determine the desired period end: if it's the last period, set to test_end; otherwise, next rebalance date.
        if i == len(rebalance_dates_sorted) - 1:
            desired_end = pd.to_datetime('2025-01-16')
        else:
            desired_end = pd.to_datetime(rebalance_dates_sorted[i+1])
        period_end = pred_index[pred_index.get_indexer([desired_end], method='bfill')[0]]
        
        # Create a daily date range for the current rebalancing period (using calendar days)
        period_days = pd.date_range(start=period_start, end=period_end, freq='D')
        
        # Get the weights for this period and compute initial allocation for each stock.
        weights = weights_per_month[rebalance_dates_sorted[i]]
        allocated_capital = {stock: weights[stock] * portfolio_value for stock in tickers}
        
        for d in period_days:
            day_stock_values = {}
            # For each stock, update the value using its predicted cumulative return
            for stock in tickers:
                # We adjust 'd' to the next available trading day if not present in predictions.
                try:
                    cum_return_day = predictions[stock]['Cumulative_Return_Strategy'].loc[d]
                except KeyError:
                    available_dates = pd.to_datetime(predictions[stock].index)
                    d_adjusted = available_dates[available_dates.get_indexer([d], method='bfill')[0]]
                    cum_return_day = predictions[stock]['Cumulative_Return_Strategy'].loc[d_adjusted]
                    
                # Get the cumulative return at the period start
                cum_return_start = predictions[stock]['Cumulative_Return_Strategy'].loc[period_start]
                
                # Compute the multiplicative growth factor for the day
                factor = cum_return_day / cum_return_start
                
                day_stock_values[stock] = allocated_capital[stock] * factor
            
            # Total Portfolio Value after a day
            daily_total = sum(day_stock_values.values())
            
            record = {'Date': d,
                      'Total_Portfolio_Value': daily_total}
            
            for stock in tickers:
                record[f'{stock}_Value'] = round(day_stock_values.get(stock, 0), 4)
                
            daily_records.append(record)
        
        # At the end of the period, update the portfolio value for the next period.
        portfolio_value = daily_records[-1]['Total_Portfolio_Value']
    
    df_daily_report = pd.DataFrame(daily_records).set_index('Date')
    return df_daily_report

# Generate and display the daily portfolio report.
portfolio_daily_report = calculate_portfolio_daily_report()
portfolio_daily_report = portfolio_daily_report[~portfolio_daily_report.index.duplicated(keep='last')]
display("Portfolio Daily Report", portfolio_daily_report)
portfolio_daily_report.to_csv("portfolio_daily_report.csv")
#%%
def calculate_equally_weighted_portfolio_report():
    portfolio_value = initial_capital
    rebalance_dates_sorted = list(weights_per_month.keys())
    daily_records = []
    
    # Index of trading dates from 1 March 2024 until 16 January 2025
    pred_index = pd.to_datetime(predictions[tickers[0]].index)

    for i in range(len(rebalance_dates_sorted)):
        desired_start = pd.to_datetime(rebalance_dates_sorted[i])
        period_start = pred_index[pred_index.get_indexer([desired_start], method='bfill')[0]]
        
        if i == len(rebalance_dates_sorted) - 1:
            desired_end = pd.to_datetime('2025-01-16')
        else:
            desired_end = pd.to_datetime(rebalance_dates_sorted[i + 1])
        period_end = pred_index[pred_index.get_indexer([desired_end], method='bfill')[0]]
        
        # Create a daily date range for the current rebalancing period (using calendar days)
        period_days = pd.date_range(start=period_start, end=period_end, freq='D')
        
        # Equally weighted portfolio
        equal_weight = 1.0 / len(tickers)
        allocated_capital = {stock: equal_weight * portfolio_value for stock in tickers}
        
        for d in period_days:
            day_stock_values = {}
            # For each stock, update the value using its predicted cumulative return
            for stock in tickers:
                try:
                    cum_return_day = predictions[stock]['Cumulative_Return_Strategy'].loc[d]
                except KeyError:
                    available_dates = pd.to_datetime(predictions[stock].index)
                    d_adjusted = available_dates[available_dates.get_indexer([d], method='bfill')[0]]
                    cum_return_day = predictions[stock]['Cumulative_Return_Strategy'].loc[d_adjusted]
                
                # Get the cumulative return at the period start
                cum_return_start = predictions[stock]['Cumulative_Return_Strategy'].loc[period_start]
                
                # Compute the multiplicative growth factor for the day
                factor = cum_return_day / cum_return_start
                day_stock_values[stock] = allocated_capital[stock] * factor
            
            # Total Portfolio Value after a day
            daily_total = sum(day_stock_values.values())
            
            record = {'Date': d, 'Total_Portfolio_Value': daily_total}
            for stock in tickers:
                record[f'{stock}_Value'] = round(day_stock_values.get(stock, 0), 4)
            
            daily_records.append(record)
        
        # At the end of the period, update the portfolio value for the next period.
        portfolio_value = daily_records[-1]['Total_Portfolio_Value']

    df_daily_report = pd.DataFrame(daily_records).set_index('Date')
    return df_daily_report

# Generate and display the daily equally weighted portfolio report.
equally_weighted_report = calculate_equally_weighted_portfolio_report()
equally_weighted_report = equally_weighted_report[~equally_weighted_report.index.duplicated(keep='last')]
display("Equally Weighted Portfolio Daily Report", equally_weighted_report)
equally_weighted_report.to_csv("equally_weighted_portfolio_report.csv")

#%%
def calculate_equally_weighted_buy_and_hold_daily_report():
    portfolio_value = initial_capital
    daily_records = []
    
    # Index of trading dates from 1 March 2024 until 16 January 2025
    pred_index = pd.to_datetime(predictions[tickers[0]].index)

    # Start and end of test period
    start_date = pd.to_datetime('2024-03-01')
    end_date = pd.to_datetime('2025-01-16')
    test_start = pred_index[pred_index.get_indexer([start_date], method='bfill')[0]]
    test_end = pred_index[pred_index.get_indexer([end_date], method='bfill')[0]]
    test_days = pd.date_range(start=test_start, end=test_end, freq='D')

    # Get start close price for each stock
    start_close = {}
    for stock in tickers:
        start_close[stock] = predictions[stock]['Close'].loc[test_start]

    capital_per_stock = portfolio_value / len(tickers)

    for d in test_days:
        day_stock_values = {}
        for stock in tickers:
            try:
                close_today = predictions[stock]['Close'].loc[d]
            except KeyError:
                available_dates = pd.to_datetime(predictions[stock].index)
                d_adjusted = available_dates[available_dates.get_indexer([d], method='bfill')[0]]
                close_today = predictions[stock]['Close'].loc[d_adjusted]
            
            # Compute the multiplicative growth factor for the day
            growth = close_today / start_close[stock]
            day_stock_values[stock] = capital_per_stock * growth

        # Total Portfolio Value after a day
        daily_total = sum(day_stock_values.values())
        record = {'Date': d, 'Total_Portfolio_Value': daily_total}

        for stock in tickers:
            record[f'{stock}_Value'] = round(day_stock_values.get(stock, 0), 4)

        daily_records.append(record)

    df_buy_hold = pd.DataFrame(daily_records).set_index('Date')
    return df_buy_hold

# Generate and export the Buy-and-Hold daily portfolio report
equally_weighted_buy_and_hold_report = calculate_equally_weighted_buy_and_hold_daily_report()
equally_weighted_buy_and_hold_report = equally_weighted_buy_and_hold_report[~equally_weighted_buy_and_hold_report.index.duplicated(keep='last')]
display("Equally Weighted Buy and Hold Daily Report", equally_weighted_buy_and_hold_report)
equally_weighted_buy_and_hold_report.to_csv("equally_weighted_buy_and_hold_daily_report.csv")

#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.plot(portfolio_daily_report.index, portfolio_daily_report['Total_Portfolio_Value'], label='Rebalanced Portfolio')
plt.plot(equally_weighted_report.index, equally_weighted_report['Total_Portfolio_Value'], label='Equally Weighted Portfolio')
plt.plot(equally_weighted_buy_and_hold_report.index, equally_weighted_buy_and_hold_report['Total_Portfolio_Value'], label='Equaly Weighted Buy and Hold Portfolio')

plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.title('Portfolio Value Over Time: Strategy Comparison')
plt.legend()
plt.tight_layout()
plt.show()

#%%
combined_returns = pd.DataFrame()
combined_returns.index = equally_weighted_buy_and_hold_report.index
combined_returns['Value_Benchmark'] = equally_weighted_buy_and_hold_report['Total_Portfolio_Value']
combined_returns['Value_EqualWeight'] = equally_weighted_report['Total_Portfolio_Value']
combined_returns['Value_RebalanceStrategy'] = portfolio_daily_report['Total_Portfolio_Value']

combined_returns
#%%
combined_returns.plot()
#%%
combined_returns.to_csv('combined_values.csv')