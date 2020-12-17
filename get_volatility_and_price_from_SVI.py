from scipy.interpolate import interp2d
from scipy import stats
from math import log, sqrt, exp, pi
import argparse
import pandas as pd
import plotly.graph_objects as go


class Calculator:
    def __init__(self, option_type, strikes, times, surface, spot, r):
        self.type = option_type
        self.strikes = strikes
        self.times = times
        self.surface = surface
        self.spot = spot
        self.r = r

    def get_volatility(self, strike, time):
        f = interp2d(self.times, [log(s / self.spot) for s in self.strikes], self.surface)
        k = log(strike / self.spot)
        return f(time, k)[0]

    def get_prices(self, strike, time, volatility):
        discount = 1 / (1 + self.r) ** time

        if self.type == 'call':
            call = self.price_by_black_scholes(strike, time, volatility)
            put = call - self.spot + discount * strike
        else:
            put = self.price_by_black_scholes(strike, time, volatility)
            call = put + self.spot - discount * strike

        return call, put

    def price_by_black_scholes(self, strike, time, volatility):
        d1 = self.d1(strike, time, volatility)
        d2 = self.d2(strike, time, volatility)

        if self.type == 'call':
            return self.spot * self.N(d1) - strike * exp(-self.r * time) * self.N(d2)
        else:
            return strike * exp(-self.r * time) * self.N(-d2) - self.spot * self.N(-d1)

    def delta(self, strike, time, volatility):
        if self.type == 'call':
            return self.N(self.d1(strike, time, volatility))
        else:
            return self.N(self.d1(strike, time, volatility)) - 1

    def gamma(self, strike, time, volatility):
        return self.N1(self.d1(strike, time, volatility)) / (self.spot * volatility * sqrt(time))

    def vega(self, strike, time, volatility):
        return self.spot * self.N1(self.d1(strike, time, volatility)) * sqrt(time)

    def theta(self, strike, time, volatility):
        if self.type == 'call':
            return - self.spot * self.N1(self.d1(strike, time, volatility)) * volatility / (2 * sqrt(time)) - \
                   self.r * strike * exp(- self.r * time) * self.N(self.d2(strike, time, volatility))
        else:
            return - self.spot * self.N1(self.d1(strike, time, volatility)) * volatility / (2 * sqrt(time)) + \
                   self.r * strike * exp(- self.r * time) * self.N(-self.d2(strike, time, volatility))

    def rho(self, strike, time, volatility):
        if self.type == 'call':
            return strike * time * exp(-self.r * time) * self.N(self.d2(strike, time, volatility))
        else:
            return - strike * time * exp(- self.r * time) * self.N(-self.d2(strike, time, volatility))

    def d1(self, strike, time, volatility):
        return (log(self.spot / strike) + (self.r + 0.5 * volatility ** 2) * time) / (volatility * sqrt(time))

    def d2(self, strike, time, volatility):
        return self.d1(strike, time, volatility) - volatility * sqrt(time)

    def N(self, x):
        return stats.norm.cdf(x)

    def N1(self, x):
        return stats.norm.pdf(x)

    def plot_surface(self, strike, time, volatility):
        fig = go.Figure()

        fig.add_surface(x=times, y=strikes, z=surface, colorbar={'thickness': 10})
        fig.add_trace(go.Scatter3d(x=[time], y=[strike], z=[volatility], mode='markers', marker=dict(size=5)))

        fig.update_layout(title='Value for input strike and time', width=900, height=900)
        fig.update_layout(scene=dict(xaxis_title='Time', yaxis_title='Strike', zaxis_title='Volatility'))
        fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get volatility and option price from SSVI')

    parser.add_argument("s", type=float, help="Spot price")
    parser.add_argument("x", type=float, help="Strike price")
    parser.add_argument("t", type=str, help="Expiration time, YYYY-mm-dd:hh:mm")
    parser.add_argument("f", type=str, help="Input file name")

    args = parser.parse_args()

    spot = args.s
    strike = args.x
    date = pd.to_datetime(args.t, format='%Y-%m-%d:%H:%M')
    file = args.f

    df1 = pd.read_csv('parametres_' + file)
    df2 = pd.read_csv(file)

    option_type = df1['Option type'].values[0]
    price_type = df1['Price type'].values[0]
    r = df1['r'].values[0]
    strikes = df2['strike'].values
    dates = df2.columns[1:]
    today = pd.to_datetime(df1['Today'].values[0], format='%Y-%m-%d %H:%M')

    if date < today or date > pd.to_datetime(dates[-1], format='%Y-%m-%d %H:%M:%S'):
        print('Date is out of limits')
    elif strike < strikes[0] or strike > strikes[-1]:
        print('Strike is out of limits')
    else:
        diff = date - today
        hours, remainder = divmod(diff.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        time = diff.days / 365 + hours / (356 * 24) + minutes / (365 * 24 * 60)

        times = []
        for day in dates:
            times.append(abs(today - pd.to_datetime(day, format='%Y-%m-%d %H:%M:%S')).days / 365)

        surface = []
        for i in range(len(strikes)):
            surface.append(df2.iloc[i, 1:].values)

        o = Calculator(option_type, strikes, times, surface, spot, r)

        volatility = o.get_volatility(strike, time)
        call, put = o.get_prices(strike, time, volatility)
        delta = o.delta(strike, time, volatility)
        gamma = o.gamma(strike, time, volatility)
        vega = o.vega(strike, time, volatility)
        theta = o.theta(strike, time, volatility)
        rho = o.rho(strike, time, volatility)

        df = pd.DataFrame({'Column': [price_type, spot, strike, date, volatility * 100, call, put,
                                      delta, gamma, vega, theta, rho]})
        df.index = ['Price type', 'Spot', 'Strike', 'Expiration time', 'Volatility', 'Call price', 'Put price',
                    'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']

        df.loc['Spot'] = df.loc['Spot'].map('${:,.2f}'.format)
        df.loc['Strike'] = df.loc['Strike'].map('${:,.2f}'.format)
        df.loc['Call price'] = df.loc['Call price'].map('${:,.2f}'.format)
        df.loc['Put price'] = df.loc['Put price'].map('${:,.2f}'.format)
        df.loc['Volatility'] = df.loc['Volatility'].map('{:,.2f}%'.format)

        print('\n', df.to_string(header=False))

        o.plot_surface(strike, time, volatility)