import streamlit as st
import pandas as pd
import numpy as np
import datetime
import markowitz_example as mwe
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Markowitz Portfolio Optimization", page_icon="🧊", layout="wide", initial_sidebar_state="expanded" )

# run_cache = {}

@st.cache
def load_data(choice):
    data = mwe.data.get_data(choice)
    return data

@st.cache
def get_opt_portfolios(choice,asset_selection):
    df = load_data(choice)
    df = df[df.index > pd.to_datetime(datetime.date(2019, 1, 1))]
    df = df[asset_selection]

    # Get Optimal PFolios:
    log_returns = df.pct_change().apply(lambda x: np.log(1 + x)).dropna()
    return_vec = log_returns.T  # returns.T
    opt_weights, opt_returns, opt_risks, res = mwe.optimal_portfolio(return_vec, N=500)

    opt_portfolios = res.copy().round(4)
    opt_portfolios.drop_duplicates(subset=["sharpe"], inplace=True)
    pfolio_by_mu = opt_portfolios.set_index('mu')

    # Generate Some Samples:
    gen_means, gen_stds = mwe.random_portfolio_mean_std(return_vec, 2000)
    scatter_stats = (opt_weights, opt_returns, opt_risks, gen_means, gen_stds)
    return df, pfolio_by_mu, scatter_stats

data_load_state = st.sidebar.text('Loading data...')
choice = 'BTC'
choice = st.sidebar.selectbox("Dataset:", ('BTC', 'ETF'))

data = load_data(choice)
asset_selection = list(data.columns)
asset_selection = st.sidebar.multiselect("Assets Included", list(data.columns), default =list(data.columns) )
data = data[data.index > pd.to_datetime(datetime.date(2019, 1, 1))]
data = data[asset_selection]

data_load_state.text("Done! (using st.cache)")
st.subheader("Market Evolution")
fig0 = px.line((data / data.iloc[0]))
st.plotly_chart(fig0, use_container_width=True)

opt_portfolios = None

st.subheader('Risk Adjusted Portfolio')

# q = risk_level.
df, pfolio_by_mu, scatter_stats = get_opt_portfolios(choice,asset_selection)

(opt_weights, opt_returns, opt_risks, gen_means, gen_stds) = scatter_stats

fig_boundary = go.Figure()
fig_boundary.add_trace(go.Scatter(x=gen_stds, y=gen_means, mode='markers', name='Sample Outcomes'))
fig_boundary.add_trace(go.Line(x=opt_risks, y=opt_returns, mode='lines+markers', name='Efficient Frontier'))
fig_boundary.layout.xaxis.title.text = "Daily Returns: Standard Deviation"
fig_boundary.layout.yaxis.title.text = "Daily Returns: Mean"

st.plotly_chart( fig_boundary, use_container_width=True )

risk_level = st.slider("Choose Risk Level", min_value=0, max_value=len(pfolio_by_mu)-1, value=0, step=1)
selected_pfolio, selection_weights = mwe.create_pfolio_from_q(pfolio_by_mu, df, len(pfolio_by_mu)-risk_level-1)
selected_pfolios = mwe.generate_pfolios(df, selection_weights)

opt_weights = pd.Series( list( np.squeeze(np.asarray(opt_weights)) ) )
opt_weights.index = selection_weights.index
best_sharpe_pfolio = mwe.generate_pfolios(df, opt_weights)
#selected_pfolios['BEST_SHARPE'] = best_sharpe_pfolio['RISK_WEIGHTED']

st.subheader("Monthly Averages")
st.dataframe( mwe.get_monthly_returns(selected_pfolios).T )
# st.dataframe( mwe.get_monthly_returns(pfolios).T )

st.subheader("Portfolio Returns")
# pfolios = mwe.generate_pfolios(df, selection_weights)
fig2 = px.line(selected_pfolios)
st.plotly_chart( fig2, use_container_width=True )

st.subheader("Optimal VAR/Return Portfolios")
st.dataframe(pfolio_by_mu)

if choice=='BTC':
    st.header("Portfolio (BTC) Stress-Testing")
    st.subheader("2018 Crash")
    df1 = load_data(choice).copy()
    btc_crash_one = df1[
        (df1.index > pd.to_datetime(datetime.date(2018, 1, 1))) & (df1.index < pd.to_datetime(datetime.date(2019, 1, 1)))]
    btc_crash_one_pfolios = mwe.generate_pfolios(btc_crash_one, selection_weights)

    fig3 = px.line(btc_crash_one_pfolios)
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(mwe.get_monthly_returns(btc_crash_one_pfolios).T)

    st.subheader("2019 Mini-Crash")
    btc_crash_two = df[
        (df.index > pd.to_datetime(datetime.date(2019, 7, 1))) & (df.index < pd.to_datetime(datetime.date(2020, 3, 1)))]
    btc_crash_two_pfolios = mwe.generate_pfolios(btc_crash_two, selection_weights)
    fig4 = px.line(btc_crash_two_pfolios)
    st.plotly_chart(fig4, use_container_width=True)
    st.dataframe(mwe.get_monthly_returns(btc_crash_two_pfolios).T)
