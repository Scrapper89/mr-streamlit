import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import skopt
import data_loading as data

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

def random_portfolio_mean_std(return_vec, nsamples):
    p = np.asmatrix(np.mean(return_vec, axis=1))
    w = np.asmatrix(rand_weights((return_vec.shape[0], nsamples))).T
    C = np.asmatrix(np.cov(return_vec))
    mu = list( np.squeeze(np.asarray((w * p.T))) )
    sigma = list( np.sqrt(np.diagonal(w * C * w.T)) )
    return mu, sigma

# def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
#     import quadprog
#     qp_G = .5 * (P + P.T)   # make sure P is symmetric
#     qp_a = -q
#     if A is not None:
#         qp_C = -np.vstack([A, G]).T
#         qp_b = -np.hstack([b, h])
#         meq = A.shape[0]
#     else:  # no equality constraint
#         qp_C = -G.T
#         qp_b = -h
#         meq = 0
#     return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def optimal_portfolio(return_vec, N = 1000):
    import cvxopt as opt
    from cvxopt import blas, solvers
    # Turn off progress printing
    solvers.options['show_progress'] = False

    n = len(return_vec)
    log_returns = np.asmatrix(return_vec)

    #min_max_return = np.round( max( [np.abs(np.max(log_returns)), np.abs(np.min(log_returns))]), 2)
    #mus = [ np.round(t/N*min_max_return,4) for t in range(1,N+1)]
    mus = [10 ** (5.0 * t / N - 2.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(log_returns))
    pbar = opt.matrix(np.mean(log_returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(S*mu, -pbar, G, h, A, b)['x']
                  for mu in mus]

    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    t_ret = blas.dot(pbar, wt)
    t_risk = np.sqrt(blas.dot(wt, S * wt))

    pfolio_weights = [np.array(x).T[0] for x in portfolios]
    res = pd.DataFrame(pfolio_weights)
    res.columns = list(return_vec.index)
    res['mu'] = mus
    res['return'] = returns
    res['risk'] = risks
    res['sharpe'] = res['return'] / np.sqrt( res['risk'] ) * np.sqrt( 252 )

    opt_weights, opt_returns, opt_risks = np.asarray(wt), returns, risks

    return np.asarray(wt), returns, risks, res

def generate_opt_portfolios(df, nrange = 2000):
    """
    w^{T}\Sigma w-q*R^{T}w
    Brute-Force Solution
    """
    #returns = np.log(df/df.shift(1)).dropna()
    # Log of percentage change
    log_returns = df.pct_change().apply(lambda x: np.log(1 + x))
    #R=(df.diff()[1:])/df.iloc[0]*100 # In percentage terms, per-day
    #R = log_returns
    #R = df.pct_change().dropna()
    #w = np.array( [1.0/len(df.columns)]*len(df.columns) )
    ER = log_returns.mean()
    #PR = (w * ER).sum()
    Sigma = log_returns.cov()
    #VAR = np.dot( np.dot( w , Sigma ) , w )

    def f(w):
        var = np.dot(np.dot(w, Sigma), w) *252
        PR = (w*ER).sum() * 252
        sharpe = PR/np.sqrt(var)
        return var, PR, 0.5*var - PR, sharpe

    # q_max = np.ceil( np.matrix(Sigma).diagonal().max() )* qMaxScale * 252
    # q_range = [x/nq*q_max for x in range(nq)]

    res = []
    for ii in range(nrange):
        t_w = rand_weights(len(df.columns))
        # for q in q_range:
        var, pr, fmin, sharpe = f(t_w)
        res.append( [fmin, var, pr, sharpe, *t_w] )

    df_res = pd.DataFrame(res)
    df_res.columns = ['fmin', 'var', 'pr', 'sharpe'] + list(df.columns)
    df_res.sort_values(by='fmin')

    df_res['pr_bucket'] = df_res.pr.round(2)
    idx_min = df_res.groupby(['pr_bucket'])['fmin'].transform(min) == df_res['fmin']
    idx_sharpe = df_res.groupby(['pr_bucket'])['sharpe'].transform(max) == df_res['sharpe']

    # idx_min = df_res.groupby(['q'])['fmin'].transform(min) == df_res['fmin']
    # idx_sharpe = df_res.groupby(['q'])['sharpe'].transform(max) == df_res['sharpe']
    opt_portfolios = df_res[idx_min]
    opt_sharpe = df_res[idx_sharpe]
    opt_portfolios = opt_portfolios.append(opt_sharpe)
    opt_portfolios = opt_portfolios[['var', 'pr','sharpe'] + list(df.columns)].drop_duplicates()
    opt_portfolios = opt_portfolios.reset_index(drop=True)

    if False:
        ax = df_res[['var', 'pr']].plot.scatter('var', 'pr')
        opt_portfolios.plot.scatter('var', 'pr', c='r', ax = ax)

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
    if isinstance(opt_portfolios,pd.DataFrame):
        for i in opt_portfolios.index:
            #risk_weights = notional_weights*opt_portfolios.iloc[i,3:] #
            risk_weights = notional_weights * opt_portfolios.loc[i, :]  #
            pfolio_rw = (df*risk_weights).sum(axis=1)
            pfolios['RW'+str(i)] = pfolio_rw
    elif isinstance(opt_portfolios,pd.Series):
        risk_weights = notional_weights * opt_portfolios
        pfolio_rw = (df * risk_weights).sum(axis=1)
        pfolios['RISK_WEIGHTED'] = pfolio_rw
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

def create_pfolio_from_q(pfolio_by_mu, df, idx):
    # assert(q>=0)
    # assert(q<=1)
    # q_max = pfolio_by_mu.index.max()
    # idx = pfolio_by_mu.index.searchsorted(q*q_max,'left')
    selection = pfolio_by_mu.iloc[idx]
    selection.drop(['risk', 'return', 'sharpe'], errors='ignore', inplace=True)
    notional_weights = 1 / df.iloc[0]
    pfolio_rw = (df * notional_weights * selection).sum(axis=1)
    return pfolio_rw, selection

def plot_sample_returns_frontier(return_vec,opt_risks,opt_returns ,n_portfolios = 10000):
    means, stds = random_portfolio_mean_std(return_vec , n_portfolios)
    # means, stds = np.column_stack([
    #     random_portfolio_mean_std(return_vec , n_portfolios)
    #     for _ in range(n_portfolios)
    # ])

    fig = plt.figure()
    plt.plot(stds, means, 'o', markersize=5)
    plt.xlabel('std')
    plt.ylabel('mean')

    plt.ylabel('mean')
    plt.xlabel('std')
    plt.plot(opt_risks, opt_returns, 'y-o')

def new_run():
    df = data.get_data(choice='BTC')
    df = df[df.index > pd.to_datetime(datetime.date(2019, 1, 1))]
    log_returns = df.pct_change().apply(lambda x: np.log(1 + x)).dropna()
    return_vec = log_returns.T  # returns.T
    # Get Optimal PFolios:
    opt_weights, opt_returns, opt_risks, res = optimal_portfolio(return_vec, N = 20)

    # Plot Sample Returns
    plot_sample_returns_frontier(return_vec, opt_risks, opt_returns, n_portfolios=500)

    opt_portfolios = res.copy().round(4)
    opt_portfolios.drop_duplicates(subset=["sharpe"], inplace=True)
    pfolio_by_mu = opt_portfolios.set_index('mu')
    pfolio_by_mu.index = pfolio_by_mu.index/pfolio_by_mu.index.max()

    selected_pfolio, selection_weights = create_pfolio_from_q(pfolio_by_mu, df, idx = 1)
    selected_pfolios = generate_pfolios(df, selection_weights)

    print(get_monthly_returns(selected_pfolios))

    # Stats for ALL:
    if False:
        pfolios = generate_pfolios(df, opt_portfolios[df.columns])
        pfolios.plot()

    # selected_pfolio = opt_portfolios[opt_portfolios.mu == opt_portfolios.mu.tail(1).values[0]][df.columns]
    # selected_pfolios = generate_pfolios(df, selected_pfolio)
    # print(get_monthly_returns(selected_pfolios))

def run():
    df = data.get_data(choice='BTC', useCache = False)
    df = df[df.index > pd.to_datetime(datetime.date(2019, 1, 1))]
    opt_portfolios = generate_opt_portfolios(df)
    pfolios = generate_pfolios(df, opt_portfolios)
    pfolios.plot()
    print(get_monthly_returns(pfolios))

# Stress-Test
def run_stress_test(data, tickers, choice):
    df = data.get_data(choice)
    btc_crash_one = df[(df.index>pd.to_datetime(datetime.date(2018,1,1)))&(df.index<pd.to_datetime(datetime.date(2019,1,1)))]
    btc_crash_two = df[(df.index>pd.to_datetime(datetime.date(2019,7,1)))&(df.index<pd.to_datetime(datetime.date(2020,3,1)))]

    btc_crash_one_pfolios = generate_pfolios(btc_crash_one,opt_portfolios)
    btc_crash_one_pfolios.plot()
    get_monthly_returns(btc_crash_one_pfolios)

    btc_crash_two_pfolios = generate_pfolios(btc_crash_two, opt_portfolios)
    btc_crash_two_pfolios.plot()
    get_monthly_returns(btc_crash_two_pfolios)
