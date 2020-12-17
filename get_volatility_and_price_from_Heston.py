import QuantLib as ql
import pandas as pd
import numpy as np
import scipy
from scipy import stats
import scipy.integrate as integrate
from numpy import exp, log, pi, sqrt, real
import plotly.graph_objects as go
import argparse
from scipy.optimize import minimize, Bounds, least_squares, root
import datetime

class Heston(ql.HestonModel):
    def __init__(self, strikes, data, calculation_date, expiration_dates, spot,  init_params=(0.1, 16., 1., 0.1, 1.)):
        self.init_params = init_params
        v0, kappa, theta, sigma, rho = self.init_params

        ql.Settings.instance().evaluationDate = calculation_date
        dividend_yield = ql.QuoteHandle(ql.SimpleQuote(0.0))
        risk_free_rate = 0.0
        dividend_rate = 0.0
        day_count = ql.Actual365Fixed()
        self.flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, risk_free_rate, day_count))
        self.dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, dividend_rate, day_count))
        self.calendar = ql.UnitedStates()

        self.process = ql.HestonProcess(self.flat_ts, self.dividend_ts, ql.QuoteHandle(ql.SimpleQuote(spot)), v0, kappa, theta, sigma,
                                   rho)
        super().__init__(self.process)

        self.engine = ql.AnalyticHestonEngine(self)
        self.vol_surf = ql.HestonBlackVolSurface(ql.HestonModelHandle(self), self.engine.Gatheral)
        data = np.array(data).T.tolist()
        self.ttm = [ql.Period(m - calculation_date, ql.Days) for m in expiration_dates]
        self.spot = spot
        self.K = strikes
        self.vol = data

        self.build_helpers()

    def build_helpers(self):
        K = self.K
        vol = self.vol
        mat = self.ttm
        spot = self.spot

        temp = []
        for m, v in zip(mat, vol):
            for i, s in enumerate(K):
                temp.append(ql.HestonModelHelper(m, self.calendar, float(spot), float(s), ql.QuoteHandle(ql.SimpleQuote(v[i])),
                                                 self.flat_ts, self.dividend_ts))
        for x in temp: x.setPricingEngine(self.engine)
        self.helpers = temp
        self.loss = [x.calibrationError() for x in self.helpers]

    def f_cost(self, params, norm=False):
        self.setParams(ql.Array(list(params)))
        self.build_helpers()
        if norm == True:
            self.loss = np.sqrt(np.sum(self.loss))
        return self.loss


class Methods:
    def scilm(self):
        root(self.f_cost, self.init_params, method='lm')


class Calculator:
    def __init__(self, option_type, strikes, times, surface, spot, r, params, today, date):
        self.type = option_type
        self.strikes = strikes
        self.times = times
        self.surface = surface
        self.spot = spot
        self.r = r
        self.params = params
        self.today = today
        self.date = date

    def get_volatility(self, strike, time, v):
        self.calculation_date = ql.Date(self.today.day, self.today.month, self.today.year)
        hours_added = datetime.timedelta(hours=8)
        hours_added_for_today = datetime.timedelta(hours=self.today.hour, minutes=self.today.minute)
        self.expiration_dates = []
        for d in self.date:
            d_ = pd.to_datetime(d, format='%d%b%y')
            self.expiration_dates.append(ql.Date(d_.day, d_.month, d_.year))

        h = Heston(strikes, v, self.calculation_date, self.expiration_dates, self.spot)

        method_dict = {'SciLM': Methods.scilm}
        for i in method_dict:
            method_dict[i](h)
            h.build_helpers()

        h.v0 = self.params[0]
        h.kappa = self.params[2]
        h.theta = self.params[1]
        h.sigma = self.params[3]
        h.rho = self.params[4]

        print(time, strike)

        return h.vol_surf.blackVol(time, strike)

    def get_prices_by_heston(self, strike, time):
        discount = 1 / (1 + self.r) ** time

        if self.type == 'call':
            call = self.price_by_heston(strike, time)
            put = call - self.spot + discount * strike
        else:
            put = self.price_by_heston(strike, time)
            call = put + self.spot - discount * strike

        return call, put

    def price_by_heston(self, strike, time):
        return self.call_heston_cf(self.spot, 0, time, strike, *self.params)

    def call_heston_cf(self, s0, r, t, k, v0, vbar, a, vvol, rho):
        def int1(w, s0, v0, vbar, a, vvol, r, rho, t, k):
            return real((exp(-1j * w * log(k)) * self.chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, w - 1j) \
                               / (1j * w * self.chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, -1j))))

        int1 = integrate.quad(lambda w: int1(w, s0, v0, vbar, a, vvol, r, rho, t, k), 0, 100)
        pi1 = int1[0] / pi + 0.5

        def int2(w, s0, v0, vbar, a, vvol, r, rho, t, k):
            return real(
                (exp(-1j * w * log(k)) * self.chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, w) / (1j * w)))

        int2 = integrate.quad(lambda w: int2(w, s0, v0, vbar, a, vvol, r, rho, t, k), 0, 100)
        pi2 = int2[0] / pi + 0.5

        y = s0 * pi1 - exp(-r * t) * k * pi2

        return y

    def chfun_heston(self, s0, v0, vbar, a, vvol, r, rho, t, w):
        alpha = -w * w / 2 - 1j * w / 2
        beta = a - rho * vvol * 1j * w
        gamma = vvol * vvol / 2
        h = sqrt(beta * beta - 4 * alpha * gamma)
        rplus = (beta + h) / vvol / vvol
        rminus = (beta - h) / vvol / vvol
        g = rminus / rplus

        C = a * (rminus * t - (2 / vvol ** 2) * log((1 - g * exp(-h * t)) / (1 - g)))
        D = rminus * (1 - exp(-h * t)) / (1 - g * exp(-h * t))

        y = exp(C * vbar + D * v0 + 1j * w * log(s0 * exp(r * t)))

        return y

    def get_prices_by_black_scholes(self, strike, time, volatility):
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
    df3 = pd.read_csv('_' + file)

    option_type = df1['Option type'].values[0]
    price_type = df1['Price type'].values[0]
    r = df1['r'].values[0]
    strikes = df2['strike'].values
    dates = df2.columns[1:]
    today = pd.to_datetime(df1['Today'].values[0], format='%Y-%m-%d %H:%M')

    params = []
    p1 = df1['v0'].dropna().values
    p2 = df1['vbar'].dropna().values
    p3 = df1['a'].dropna().values
    p4 = df1['vvol'].dropna().values
    p5 = df1['rho'].dropna().values
    for i in range(len(p1)):
        params.append([p1[i], p2[i], p3[i], p4[i], p5[i]])

    if date < today or date > pd.to_datetime(dates[-1], format='%Y-%m-%d %H:%M'):
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

        index = 0
        for i in range(len(times) - 1):
            if times[i] <= time < times[i+1]:
                index = i
                break

        surface = []
        for i in range(len(strikes)):
            surface.append(df2.iloc[i, 1:].values)

        v = []
        for i in range(len(strikes)):
            v.append(df3.iloc[i, 1:].values)

        o = Calculator(option_type, strikes, times, surface, spot, r, params[index], today, df1['Date'].dropna().values)

        volatility = o.get_volatility(strike, time, v)
        call, put = o.get_prices_by_heston(strike, time)
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
