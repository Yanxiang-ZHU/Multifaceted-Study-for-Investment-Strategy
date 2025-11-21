import numpy as np
from scipy.optimize import minimize
import pandas as pd
from scipy.optimize import basinhopping
from multiprocessing import Pool, current_process

# example data
np.random.seed(42)
portfolios = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3])
transaction_fee = 0.001
risk_threshold = 1e10   # different param for different risk management methods

# initialization
current_cash = 10000000 
# current_holdings = np.zeros(21, dtype=float)
current_holdings = np.full(21, 100.0)
# current_portfolio_holdings = np.zeros(4, dtype=float)

data = { 'day': [0],  'cash amount ($)': [100000000],  'portfolio weight 0 (%)': [0.], 'portfolio weight 1 (%)': [0.],'portfolio weight 2 (%)': [0.], 'portfolio weight 3 (%)': [0.]}
for i in range(21): data[f'stock holdings {i} (share)'] = [0]
data['expected net values($)'] = [100000000]
data['actual net values($)'] = [100000000]
df = pd.DataFrame(data)

# read the real time data
def real_time_data(n): # n should start from 15
    prediction_df = pd.read_csv('stock_return_prediction.csv')
    actual_df = pd.read_excel('stock_return_actual.xlsx')
    price_df = pd.read_excel('stock_daily_data.xlsx')

    expected_returns = prediction_df.iloc[n-1, 1:].values.T
    returns = actual_df.iloc[n-15:n-1, 1:].values.T
    real_returns = actual_df.iloc[n-1, 1:].values.T
    cov_matrix = np.cov(returns)

    columns_to_read = list(range(1, 102, 5)) 
    rows_to_read = list(range(n-15, n-1)) 
    stock_prices = price_df.iloc[rows_to_read, columns_to_read].values.T

    return stock_prices, expected_returns, cov_matrix, real_returns

# RSI
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gains = delta.clip(min=0)
    losses = -delta.clip(max=0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

# only in case of buying
def calculate_transaction_cost(new_holdings, current_holdings, prices, transaction_fee):
    buy_amounts = (new_holdings - current_holdings) * prices
    buy_amounts[buy_amounts < 0] = 0  
    return transaction_fee * np.sum(buy_amounts)

# regard variance as risk (need change here)
def calculate_return_and_risk(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return portfolio_return, portfolio_variance

# RSI constraints
def rsi_constraints(params, current_holdings, prices, use_RSI=True):
    constraints_list = []
    stock_weights = params[5:]
    total_investment = np.sum(current_holdings * prices)
    k = 0
    
    if use_RSI == True:
        for i in range(4):
            indices = np.where(portfolios == i)[0]
            for j, idx in enumerate(indices):
                rsi = calculate_rsi(stock_prices[idx])
                lower_bound = np.tanh((50 - rsi) / 30) / 2 - 0.5
                upper_bound = np.tanh((50 - rsi) / 30) / 2 + 0.5
                current_weight = (current_holdings[idx] * prices[idx]) / (total_investment + 0.01)  # Update current_weight calculation  / avoid NAN result
                new_weight = stock_weights[k]
                k += 1
                diff_ratio = (new_weight - current_weight) / current_weight if current_weight != 0 else new_weight
                constraints_list.append(lower_bound - diff_ratio)  # lower bound constraint
                constraints_list.append(diff_ratio - upper_bound)  # upper bound constraint
    
    return constraints_list

def objective(params, cov_matrix, current_holdings, prices, transaction_fee, expected_returns):
    # cash_weight = params[0] 
    portfolio_weights = params[1:5]  
    stock_weights = params[5:]

    total_value = current_cash + np.sum(current_holdings * prices)
    # cash = total_value * cash_weight  # cash
    # investable_value = total_value - cash  # for investment

    # portfolio return and risk
    portfolio_returns = []
    portfolio_variances = []
    transaction_costs = 0

    for i in range(4):
        indices = np.where(portfolios == i)[0]
        if i == 0: portfolio_stock_weights = stock_weights[0:8]
        elif i == 1: portfolio_stock_weights = stock_weights[8:10]
        elif i == 2: portfolio_stock_weights = stock_weights[10:13]
        elif i == 3: portfolio_stock_weights = stock_weights[13:21]
        portfolio_return, portfolio_variance = calculate_return_and_risk(portfolio_stock_weights, expected_returns[indices], cov_matrix[np.ix_(indices, indices)])
        new_stock_holdings = total_value * portfolio_weights[i] * stock_weights[i] / prices[indices]  # share
        
        transaction_cost = 0
        for index in indices:
            transaction_cost += calculate_transaction_cost(new_stock_holdings, current_holdings[index], prices[index], transaction_fee)
        
        portfolio_returns.append(portfolio_return)
        portfolio_variances.append(portfolio_variance)
        transaction_costs += transaction_cost

    total_return = total_value * np.dot(portfolio_weights, portfolio_returns) - transaction_costs
    total_variance = np.dot(portfolio_weights.T, np.dot(np.diag(portfolio_variances), portfolio_weights))
    risk = total_variance

    return total_return, risk

def wrapped_objective(params, cov_matrix, current_holdings, prices, transaction_fee, risk_threshold, expected_returns):
    params = projection(params)
    total_return, risk = objective(params, cov_matrix, current_holdings, prices, transaction_fee, expected_returns)
    # risk here need further changing (risk and risk_threshold)
    return -total_return


def projection(params):
    params[0:5] = np.maximum(params[0:5], 0)
    params[0:5] = params[0:5] / np.sum(params[0:5])
    
    params[5:13] = np.maximum(params[5:13], 0)
    params[5:13] = params[5:13] / np.sum(params[5:13])
    
    params[13:15] = np.maximum(params[13:15], 0)
    params[13:15] = params[13:15] / np.sum(params[13:15])
    
    params[15:18] = np.maximum(params[15:18], 0)
    params[15:18] = params[15:18] / np.sum(params[15:18])
    
    params[18:26] = np.maximum(params[18:26], 0)
    params[18:26] = params[18:26] / np.sum(params[18:26])
    
    return params



if __name__ == "__main__":
    for day in range(15,119):
        stock_prices, expected_returns, cov_matrix, real_returns = real_time_data(day)
        
        # in the first day, we suppose holding very stocks 100 shares (so we won't get into divide-zero problems)
        if day == 15:       
            current_cash = current_cash - np.sum(current_holdings * stock_prices[:, -1])
        total_value = current_cash + np.sum(current_holdings * stock_prices[:, -1])
        expected_today_profit = 0

        # the params' boundary and intial value
        current_portfolio_holdings = [np.sum(current_holdings[0:8] * stock_prices[0:8, -1]), np.sum(current_holdings[8:10] * stock_prices[8:10, -1]), np.sum(current_holdings[10:13] * stock_prices[10:13, -1]), np.sum(current_holdings[13:21] * stock_prices[13:21, -1])] / total_value


        bounds = [(0, 1)] + [(0, 1) for _ in range(4)] + [(0, 1) for _ in range(21)]  
        initial_params = np.concatenate(([current_cash/ total_value], current_portfolio_holdings, current_holdings[0:8]*stock_prices[0:8,-1] / (total_value*current_portfolio_holdings[0]), current_holdings[8:10]*stock_prices[8:10,-1] / (total_value*current_portfolio_holdings[1]), current_holdings[10:13]*stock_prices[10:13,-1] / (total_value*current_portfolio_holdings[2]), current_holdings[13:21]*stock_prices[13:21,-1] / (total_value*current_portfolio_holdings[3]))) 
        
        # use_RSI = day > 20

        # constraints = [
        #     {'type': 'ineq', 'fun': lambda params: rsi_constraints(params, current_holdings, stock_prices[:, -1], use_RSI=True)}  
        # ]


        constraints = [
            {'type': 'eq', 'fun': lambda params: np.sum(params[1:5]) + params[0] - 1},  # cash weights + stock weights should be 1 exactly
            {'type': 'eq', 'fun': lambda params: np.sum(params[5:13]) - 1 },  # stock percentage in one portfolio should be 1
            {'type': 'eq', 'fun': lambda params: np.sum(params[13:15]) - 1},
            {'type': 'eq', 'fun': lambda params: np.sum(params[15:18]) - 1},
            {'type': 'eq', 'fun': lambda params: np.sum(params[18:26]) - 1},
            {'type': 'ineq', 'fun': lambda params:  - objective(params, cov_matrix, current_holdings, stock_prices[:, -1], transaction_fee, expected_returns)[1]},  # the risk should be lower than risk threshold
            {'type': 'ineq', 'fun': lambda params: rsi_constraints(params, current_holdings, stock_prices[:, -1], use_RSI=True)}  # RSI constraints: avoiding too violent buying or selling behavior
        ]

        # def minimize_with_constraints(params):
        #     return minimize(wrapped_objective, params, args=(cov_matrix, current_holdings, stock_prices[:, -1], transaction_fee, risk_threshold, expected_returns), method='SLSQP', bounds=bounds, constraints=constraints).fun
        
        # result = basinhopping(minimize_with_constraints, initial_params, niter=1, T=0.1, stepsize=0.5)
        # optimal_params = result.x

        result = minimize(wrapped_objective, initial_params, args=(cov_matrix, current_holdings, stock_prices[:, -1], transaction_fee, risk_threshold, expected_returns), method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_params = result.x

        optimal_cash_weight = optimal_params[0]
        optimal_cash = optimal_cash_weight * total_value
        optimal_portfolio_weights = optimal_params[1:5]
        optimal_stock_weights = optimal_params[5:]

        investable_value = total_value - optimal_cash  # available for investment
        current_portfolio_holdings = optimal_portfolio_weights
        current_cash = optimal_cash

        # output print out
        new_row = {
            'day': day, 'cash amount ($)': optimal_cash, 
            'portfolio weight 0 (%)': optimal_portfolio_weights[0], 
            'portfolio weight 1 (%)': optimal_portfolio_weights[1],
            'portfolio weight 2 (%)': optimal_portfolio_weights[2],
            'portfolio weight 3 (%)': optimal_portfolio_weights[3]
        }

        expected_net_values = optimal_cash
        actual_net_values = optimal_cash
        current_holdings = []
        for i in range(21): 
            if i < 8:
                index = 0
            elif i < 10:
                index = 1
            elif i < 13:
                index = 2
            elif i < 21:
                index = 3
            new_row[f'stock holdings {i} (share)'] = total_value * optimal_portfolio_weights[index] * optimal_params[i+5] / stock_prices[i][-1]
            current_holdings.append(total_value * optimal_portfolio_weights[index] * optimal_params[i+5]/ stock_prices[i][-1])
            expected_net_values += total_value * optimal_portfolio_weights[index] * optimal_params[i+5] * (1 + expected_returns[i])
            actual_net_values += total_value * optimal_portfolio_weights[index] * optimal_params[i+5] * (1 + real_returns[i])
        new_row['expected net values($)'] = expected_net_values
        new_row['actual net values($)'] = actual_net_values
        # save the data into dataframe
        df.loc[len(df)] = new_row

    df.to_excel('trading_strategy.xlsx', index=False)

        