import pandas_datareader.data as web
import os
import requests
import json
import datetime
import pandas as pd

def get_gdrive_data(tp = 'btc'):
    if tp == 'xrp':
        link = r"https://drive.google.com/u/0/uc?id=1cx0wSRPKpsz3pDLQYb7ZG6foBETPAaFZ&export=download"
    if tp == 'btc':
        link = r"https://drive.google.com/u/0/uc?id=1EiyTdW8tKBJHURJ_OWyct3IqvHQblS0A&export=download"
    return pd.read_csv(link)

def get_investpy_data(ticker, rtype, country):
    import investpy
    countries = investpy.get_etf_countries()

    investpy.get_etf_historical_data("LOCK.UK", 'united kingdom', '01/01/2021', '23/03/2021', stock_exchange=None, as_json=False, order='ascending', interval='Daily')

    df = investpy.get_stock_historical_data(stock='AAPL',
                                            country='United States',
                                            from_date='01/01/2021',
                                            to_date='23/03/2021')

    data = investpy.get_crypto_historical_data(crypto='bitcoin',
                                               from_date='01/01/2014',
                                               to_date='01/01/2019')

    return df.head()

def get_bitmex_data(ticker = 'XBTUSD', time_ago = datetime.date(2021,3,20), count = 10):
    #from bitmex import bitmex
    #bitmex_api_key = '6ZaQ_yNAkJ6nt51XePmrXBCA'
    #bitmex_api_secret = 'k8gNwzqyFWmhy3HgMHc3-zeb9eGBtgK5or0GiI6rgJeJvlUI'

    ### CONSTANTS
    binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
    baseURI = "https://www.bitmex.com/api/v1"
    endpoint = "/trade/bucketed"
    request = requests.get(baseURI + endpoint,
                           params={'binSize': '1d', 'symbol': ticker, 'count': count, 'startTime': time_ago})
    res = json.loads(request.content)
    return pd.DataFrame(res)

def get_stooq_data( ticker ):
    return web.DataReader(ticker, 'stooq')

yahoo_tickers = {
    'BTC' : 'BTC-USD',
    'ETH' : 'ETH-USD',
    'LTC' : 'LTC-USD',
    'XRP' : 'XRP-USD'
}

def get_data_from_yahoo( ticker ):
    ticker = yahoo_tickers.get(ticker,ticker)
    return web.DataReader(ticker, data_source='yahoo')

def get_data(choice, useCache = True):
    path = os.path.join(os.getcwd(),'cache',choice+'.csv')
    if useCache and os.path.exists(path):
        df =pd.read_csv(path)
        df['Date'] = df['Date'].map(pd.to_datetime)
        df.set_index('Date', inplace=True)
        return df
    if choice == 'Stocks':
        tickers = ['^DJI', 'TSLA', 'AAPL', 'JPM','DAX']
    elif choice == 'ETF':
        tickers = ['LOCK.UK', 'INRG.UK', 'DGTL.UK', 'ECAR.UK', 'IQQQ.DE', 'HEAL.UK','2B76.DE','XASX.UK','IDAP.UK']
        #yahoo_tickers = ['LOCK.L', 'INRG.L', 'DGTL.UK', 'ECAR.L', 'IQQQ.DE', 'HEAL.L','2B76.DE']
    elif choice == 'BTC':
        tickers = ['BTC','XRP','LTC','ETH','ADA'] # 'BTC.V', 'XRP.V' 'DOGE'
        cwd = os.getcwd()
        btc = pd.read_csv(r"https://stooq.com/q/d/l/?s=btc.v&i=d")
        btc['Date'] = btc['Date'].map(pd.to_datetime)
        btc.set_index('Date',inplace=True)

        xrp = pd.read_csv(r"https://stooq.com/q/d/l/?s=xrp.v&i=d")
        xrp['Date'] = xrp['Date'].map(pd.to_datetime)
        xrp.set_index('Date',inplace=True)

        ltc = pd.read_csv(r"https://stooq.com/q/d/l/?s=ltc.v&i=d")
        ltc['Date'] = ltc['Date'].map(pd.to_datetime)
        ltc.set_index('Date', inplace=True)

        eth = pd.read_csv(r"https://stooq.com/q/d/l/?s=eth.v&i=d")
        eth['Date'] = eth['Date'].map(pd.to_datetime)
        eth.set_index('Date', inplace=True)

        ADA = pd.read_csv(r"https://stooq.com/q/d/l/?s=ada.v&i=d")
        ADA['Date'] = ADA['Date'].map(pd.to_datetime)
        ADA.set_index('Date', inplace=True)

    if choice == 'BTC':
        data = [ btc, xrp, ltc, eth, ADA ]
    else:
        data = [ web.DataReader(x, 'stooq') for x in tickers ]

    df = pd.concat( [ datai.Close for datai in data] , axis = 1 )
    df.columns = tickers
    df = df.dropna().sort_index()
    df.to_csv( path )
    return df