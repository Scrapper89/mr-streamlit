import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import pandas_datareader.data as web
import pandas_datareader.stooq as stooq
import skopt
import os

np.random.seed(123)
pd.options.display.width = 0
gen = skopt.sampler.sobol.Sobol()

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    #k = np.random.rand(n)
    k = np.random.uniform(0,4,size=n)
    return k / sum(k)

def rand_sobol(n):
    k = np.array( gen.generate(dimensions = ([(0.,1.)]*n), n_samples=1, random_state=None)[0] )
    return k / sum(k)

def get_gdrive_data(tp = 'btc'):
    if tp == 'xrp':
        link = r"https://drive.google.com/u/0/uc?id=1cx0wSRPKpsz3pDLQYb7ZG6foBETPAaFZ&export=download"
    if tp == 'btc':
        link = r"https://drive.google.com/u/0/uc?id=1EiyTdW8tKBJHURJ_OWyct3IqvHQblS0A&export=download"
    return pd.read_csv(link)

def get_data(choice):

    if choice == 'Stocks':
        tickers = ['^DJI', 'TSLA', 'AAPL', 'JPM','DAX']
    elif choice == 'ETF':
        tickers = ['LOCK.UK', 'INRG.UK', 'DGTL.UK', 'ECAR.UK', 'IQQQ.DE', 'HEAL.UK','2B76.DE']
    elif choice == 'BTC':
        tickers = ['BTC','XRP','LTC','ETH'] # 'BTC.V', 'XRP.V' 'DOGE'
        cwd = os.getcwd()
        #btc = pd.read_csv(os.path.join(cwd,"database","btc.v.csv"))
        btc = pd.read_csv(r"https://stooq.com/q/d/l/?s=btc.v&i=d")
        #btc = get_gdrive_data('btc')
        btc['Date'] = btc['Date'].map(pd.to_datetime)
        btc.set_index('Date',inplace=True)

        #xrp = pd.read_csv(os.path.join(cwd,"database","xrp.v.csv"))
        xrp = pd.read_csv(r"https://stooq.com/q/d/l/?s=xrp.v&i=d")
        #xrp = get_gdrive_data('xrp')
        xrp['Date'] = xrp['Date'].map(pd.to_datetime)
        xrp.set_index('Date',inplace=True)

        # ltc = pd.read_csv(os.path.join(cwd, "database", "ltc.v.csv"))
        # xrp = get_gdrive_data('xrp')
        ltc = pd.read_csv(r"https://stooq.com/q/d/l/?s=ltc.v&i=d")
        ltc['Date'] = ltc['Date'].map(pd.to_datetime)
        ltc.set_index('Date', inplace=True)

        #eth = pd.read_csv(os.path.join(cwd, "database", "eth.v.csv"))
        eth = pd.read_csv(r"https://stooq.com/q/d/l/?s=eth.v&i=d")
        # xrp = get_gdrive_data('xrp')
        eth['Date'] = eth['Date'].map(pd.to_datetime)
        eth.set_index('Date', inplace=True)
    if choice == 'BTC':
        data = [ btc, xrp, ltc, eth ]
    else:
        data = [ web.DataReader(x, 'stooq') for x in tickers ]

    df = pd.concat( [ datai.Close for datai in data] , axis = 1 )
    df.columns = tickers
    df = df.dropna().sort_index()
    return df


def random_portfolio(return_vec):
    '''
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(return_vec, axis=1))
    w = np.asmatrix(rand_weights(return_vec.shape[0]))
    C = np.asmatrix(np.cov(return_vec))
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    return mu, sigma

def optimal_portfolio(return_vec):
    import cvxopt as opt
    from cvxopt import blas, solvers
    # Turn off progress printing
    solvers.options['show_progress'] = False

    n = len(return_vec)
    returns = np.asmatrix(return_vec)

    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

def generate_opt_portfolios(df,nq = 100, nrange = 2000, qMaxScale = 2):
    """
    w^{T}\Sigma w-q*R^{T}w
    Brute-Force Solution
    """
    returns = np.log(df/df.shift(1)).dropna()
    # Log of percentage change
    log_returns = df.pct_change().apply(lambda x: np.log(1 + x))
    R=(df.diff()[1:])/df.iloc[0]*100 # In percentage terms, per-day
    R = df.pct_change().dropna()
    w = np.array( [1.0/len(df.columns)]*len(df.columns) )
    ER = R.mean()
    PR = (w * ER).sum()
    Sigma = R.cov()
    VAR = np.dot( np.dot( w , Sigma ) , w )

    def f(w,q):
        var = np.dot(np.dot(w, Sigma), w) *252
        PR = (w*ER).sum() * 252
        sharpe = PR/np.sqrt(var)
        return var, PR, var - q*PR, sharpe

    q_max = np.ceil( np.matrix(Sigma).diagonal().max() )* qMaxScale * 252
    q_range = [x/nq*q_max for x in range(nq)]

    res = []
    for ii in range(nrange):
        t_w = rand_weights(len(df.columns))
        for q in q_range:
            var, pr, fmin, sharpe = f(t_w,q)
            res.append( [fmin, q, var, pr, sharpe, *t_w] )

    df_res = pd.DataFrame(res)
    df_res.columns = ['fmin','q', 'var', 'pr','sharpe'] + list(df.columns)
    df_res.sort_values(by='fmin')

    df_res[['var','pr']].plot.scatter('var','pr')

    idx_min = df_res.groupby(['q'])['fmin'].transform(min) == df_res['fmin']
    idx_sharpe = df_res.groupby(['q'])['sharpe'].transform(max) == df_res['sharpe']
    opt_portfolios = df_res[idx_min]
    opt_sharpe = df_res[idx_sharpe]
    opt_portfolios = opt_portfolios.append(opt_sharpe)
    opt_portfolios = opt_portfolios[['var', 'pr','sharpe'] + list(df.columns)].drop_duplicates()
    opt_portfolios = opt_portfolios.reset_index(drop=True)
    return opt_portfolios

### BackTest
def generate_pfolios(df,opt_portfolios):
    # Notional
    pfolios = df.copy()
    pfolios = pfolios/pfolios.iloc[0]
    notional_weights = 1/df.iloc[0]
    eqw = 1/float(len(notional_weights))
    pfolio_EQW = (df*notional_weights*eqw).sum(axis=1)
    pfolios['EQUAL_W'] = pfolio_EQW

    # Risk-Weighted
    for i in opt_portfolios.index:
        risk_weights = notional_weights*opt_portfolios.iloc[i,3:]
        pfolio_rw = (df*risk_weights).sum(axis=1)
        pfolios['RW'+str(i)] = pfolio_rw
    return pfolios

# Monthly Returns
def get_monthly_returns(pfolios):
    monthly = pfolios.resample("M").last()
    gains = monthly.diff()
    means = gains.mean()
    std = gains.std()
    sharpe = means/std
    previous_peaks = pfolios.cummax()
    drawdown = ((pfolios - previous_peaks)/previous_peaks).min()*100
    round(drawdown.min(),4)
    monthly_stats = pd.DataFrame( [means, std, sharpe, drawdown ]).round(4)
    monthly_stats.index = ['MeanGain','STD','Sharpe','MaxDrawdown']
    return monthly_stats

def new_run():
    df = get_data(choice='BTC')
    df = df[df.index > pd.to_datetime(datetime.date(2019, 1, 1))]
    log_returns = df.pct_change().apply(lambda x: np.log(1 + x)).dropna()
    n_portfolios = 5000
    return_vec = log_returns.T  # returns.T
    means, stds = np.column_stack([
        random_portfolio(return_vec)
        for _ in range(n_portfolios)
    ])

    fig = plt.figure()
    plt.plot(stds, means, 'o', markersize=5)
    plt.xlabel('std')
    plt.ylabel('mean')

    opt_weights, opt_returns, opt_risks = optimal_portfolio(return_vec)
    plt.ylabel('mean')
    plt.xlabel('std')
    plt.plot(opt_risks, opt_returns, 'y-o')


def run():
    df = get_data(choice='BTC')
    df = df[df.index > pd.to_datetime(datetime.date(2019, 1, 1))]
    opt_portfolios = generate_opt_portfolios(df)
    pfolios = generate_pfolios(df, opt_portfolios)
    (pfolios - pfolios.iloc[0]).plot()
    print(get_monthly_returns(pfolios))

# Stress-Test
def run_stress_test(data, tickers, choice):
    df = get_data(choice)
    btc_crash_one = df[(df.index>pd.to_datetime(datetime.date(2018,1,1)))&(df.index<pd.to_datetime(datetime.date(2019,1,1)))]
    btc_crash_two = df[(df.index>pd.to_datetime(datetime.date(2019,7,1)))&(df.index<pd.to_datetime(datetime.date(2020,3,1)))]

    btc_crash_one_pfolios = generate_pfolios(btc_crash_one,opt_portfolios)
    btc_crash_one_pfolios.plot()
    get_monthly_returns(btc_crash_one_pfolios)

    btc_crash_two_pfolios = generate_pfolios(btc_crash_two, opt_portfolios)
    btc_crash_two_pfolios.plot()
    get_monthly_returns(btc_crash_two_pfolios)
