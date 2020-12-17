import pandas as pd
from datetime import datetime
import numpy as np
import scipy.integrate as integrate
import scipy
from numpy import exp, log, pi, sqrt
from scipy import stats
from scipy.optimize import minimize, Bounds, least_squares, root
from scipy.interpolate import CubicSpline as spl, PchipInterpolator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import datetime
import warnings
import QuantLib as ql

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


class Price:
    def __init__(self, type):
        if type == 'ask_c' or type == 'bid_c' or type == 'mid_c':
            self.type = 'call'
        else:
            self.type = 'put'


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


class Volatility:

    def __init__(self, type, date):
        self.df = pd.DataFrame()
        self.S = 0
        pr = Price(type)
        self.option_type = pr.type
        self.price_type = type
        self.date = date
        self.time = []
        self.df_params = pd.DataFrame(columns=['v0', 'vbar', 'a', 'vvol', 'rho'])
        # hours = 8
        # hours_added = datetime.timedelta(hours = hours)
        # self.today = pd.to_datetime(day, format='%Y-%m-%d:%H:%M')
        # for d in self.date:
        #     if self.today > (pd.to_datetime(d, format='%d%b%y') + hours_added):
        #         print("Date is incorrect")
        #     self.time.append(abs(self.today - (pd.to_datetime(d, format='%d%b%y') + hours_added)).days / 365)

        hours = 8
        df1 = pd.read_csv(self.date[0] + '.csv')
        day = df1['q'].values[0][:-2]
        self.today = pd.to_datetime(day, format='%Y-%m-%d %H:%M')
        self.calculation_date = ql.Date(self.today.day, self.today.month, self.today.year)
        hours_added = datetime.timedelta(hours=hours)
        hours_added_for_today = datetime.timedelta(hours=self.today.hour, minutes=self.today.minute)
        self.expiration_dates = []
        for d in self.date:
            d_ = pd.to_datetime(d, format='%d%b%y')
            if self.today + hours_added_for_today > (d_+ hours_added):
                print("Date is incorrect")
            diff = abs(self.today - d_ - hours_added)
            self.expiration_dates.append(ql.Date(d_.day, d_.month, d_.year))
            hours, remainder = divmod(diff.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            self.time.append(diff.days / 365 + hours / (365 * 24) + minutes / (365 * 24 * 60))

    def get_data_from_file(self):
        count = 0
        S = []
        for d in self.date:
            df1 = pd.read_csv(d + '.csv')
            S.append(df1['s0'][1])
            if self.price_type == 'mid_c':
                df1['mid_c'] = (df1['bid_c'] + df1['ask_c']) / 2.
            elif self.price_type == 'mid_p':
                df1['mid_p'] = (df1['bid_p'] + df1['ask_p']) / 2.
            selected_columns = df1[['k', self.price_type]]
            df1 = selected_columns.copy()
            df1 = df1.rename(columns={self.price_type: 'P' + str(count), 'k': 'strike'})
            try:
                self.df = pd.merge(self.df, df1, how='inner', on=['strike'])
            except:
                self.df = df1
            count += 1
        self.S = np.mean(S)
        self.df.dropna(inplace=True)
        self.df = self.df.reset_index(drop=True)
        self.df['k'] = self.df.apply(lambda x: log(x['strike'] / self.S), axis=1)

    def N(self, x):
        return stats.norm.cdf(x)

    def Vega(self, S_, d1, expiry):
        vega = S_ * sqrt(1. * expiry / pi / 2.) * exp(-0.5 * d1 ** 2)
        return vega

    def C_(self, d1, d2, E, r, expiry):
        if self.option_type == 'call':
            return self.S * self.N(d1) - E * exp(-r * (expiry)) * self.N(d2)
        else:
            return -self.S * self.N(-d1) + E * exp(-r * (expiry)) * self.N(-d2)

    def sigma_from_price(self, C, E, expiry, r, error):
        sigma = 1.
        dv = error + 1.
        while abs(dv) > error:
            d1 = (log(self.S / E) + (r + sigma ** 2 / 2.) * expiry) / (sigma * sqrt(expiry))
            d2 = d1 - sigma * sqrt(expiry)
            price = self.C_(d1, d2, E, r, expiry)
            vega = self.Vega(self.S, d1, expiry)
            price_error = price - C
            dv = 1. * price_error / vega
            sigma = sigma - dv
        return sigma 

    def count_vol(self):
        v = np.zeros((len(self.df), len(self.date)))
        for i in range(len(self.df)):
            for j in range(len(self.date)):
                v[i][j] = self.sigma_from_price(self.df.loc[i, 'P' + str(j)], self.df.loc[i, 'strike'], self.time[j],
                                                0., 0.001)
        return v

    def d1(self, X, T, sigma):
        r = 0
        return (np.log(self.S/X) + (r + sigma**2/2.)*T)/(sigma*(T**0.5))

    def Black_Scholes(self, X, T, sigma):
        r = 0
        d1 = (np.log(self.S/X) + (r + sigma**2/2.)*T)/(sigma*(T**0.5))
        d2 = d1 - sigma*(T**0.5)
        if self.option_type == 'call':
            C = self.S*self.N(d1)-X*np.exp(-r*T)*self.N(d2)
            return C
        else:
            P = X*np.exp(-r*T)*self.N(-d2)-self.S*self.N(-d1)
            return P

    def count_heston_volatility(self, v, h_price):
        strikes = list(self.df['strike'].values)
        bs_price = np.zeros((len(strikes), len(self.time)))
        vol = np.zeros((len(strikes), len(self.time)))

        h = Heston(strikes, v, self.calculation_date, self.expiration_dates, self.S)

        method_dict = {'SciLM': Methods.scilm}
        for i in method_dict:
            method_dict[i](h)
            h.build_helpers()

        for i in range(len(self.df_params)):
            h.v0 = self.df_params.iloc[i, 0]
            h.kappa = self.df_params.iloc[i, 2]
            h.theta = self.df_params.iloc[i, 1]
            h.sigma = self.df_params.iloc[i, 3]
            h.rho = self.df_params.iloc[i, 4]

            Z1 = np.array([h.vol_surf.blackVol(float(self.time[i]), float(x)) for x in strikes])
            vol[:, i] = Z1

        # print(h.vol_surf.blackVol(0.019178082191780823, 17000.0))
        return vol

    # def count_heston_volatility(self):
    #     strikes = list(self.df['strike'].values)
    #
    #     today = ql.Date(self.today.day, self.today.month, self.today.year)
    #
    #     day_count = ql.Actual365Fixed()
    #     spot_quote = ql.QuoteHandle(ql.SimpleQuote(self.S))
    #
    #     risk_free_curve = ql.FlatForward(today, 0, day_count)
    #     flat_ts = ql.YieldTermStructureHandle(risk_free_curve)
    #     dividend_ts = ql.YieldTermStructureHandle(risk_free_curve)
    #
    #     vol = np.zeros((len(strikes), len(self.time)))
    #     for i in range(len(self.df_params)):
    #         v0 = self.df_params.iloc[i, 0]
    #         kappa = self.df_params.iloc[i, 2]
    #         theta = self.df_params.iloc[i, 1]
    #         sigma = self.df_params.iloc[i, 3]
    #         rho = self.df_params.iloc[i, 4]
    #
    #         heston_process = ql.HestonProcess(flat_ts, dividend_ts, spot_quote, v0, kappa, theta, sigma, rho)
    #         heston_model = ql.HestonModel(heston_process)
    #
    #         heston_handle = ql.HestonModelHandle(heston_model)
    #         heston_vol_surface = ql.HestonBlackVolSurface(heston_handle)
    #
    #         Z1 = np.array([heston_vol_surface.blackVol(float(self.time[i]), float(x)) for x in strikes])
    #         vol[:, i] = Z1
    #
    #     return vol

    def costf(self, x, j, t):
        cost = []
        p = self.df[['P0', 'P1', 'P2', 'P3']].values
        for i, s in enumerate(self.df['strike'].values):
            cost.append(p[i, j] - self.call_heston_cf(
                self.S, 0, t, s, x[0], x[1], (x[4] + x[2] ** 2) / (2 * x[1]), x[2], x[3]))
        return cost

    def count_heston_price(self):
        x0 = [.5, .5, 1, -0.5, 1]
        lb = [0, 0, 0, -1, 0]
        ub = [1, 1, 100, 1, 100]

        strikes = list(self.df['strike'].values)
        z = np.zeros((len(strikes), len(self.time)))

        for i, t in enumerate(self.time):
            new_params = least_squares(self.costf, x0, args=(i, t), bounds=(lb, ub)).x

            v0 = new_params[0]
            vbar = new_params[1]
            a = (new_params[4] + new_params[2]**2)/(2*new_params[1])
            vvol = new_params[2]
            rho = new_params[3]

            self.df_params.loc[i] = {'v0': v0,
                                     'vbar': vbar,
                                     'a': a,
                                     'vvol': vvol,
                                     'rho': rho}

            z[:, i] = [self.call_heston_cf(self.S, 0, t, st, v0, vbar, a, vvol, rho) for st in strikes]
        print(self.df_params)
        return z

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

    def call_heston_cf(self, s0, r, t, k, v0, vbar, a, vvol, rho):
        def int1(w, s0, v0, vbar, a, vvol, r, rho, t, k):
            return scipy.real((exp(-1j * w * log(k)) * self.chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, w - 1j) \
                               / (1j * w * self.chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, -1j))))

        int1 = integrate.quad(lambda w: int1(w, s0, v0, vbar, a, vvol, r, rho, t, k), 0, 100)
        pi1 = int1[0] / pi + 0.5

        def int2(w, s0, v0, vbar, a, vvol, r, rho, t, k):
            return scipy.real((exp(-1j * w * log(k)) * self.chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, w) / (1j * w)))

        int2 = integrate.quad(lambda w: int2(w, s0, v0, vbar, a, vvol, r, rho, t, k), 0, 100)
        pi2 = int2[0] / pi + 0.5

        y = s0 * pi1 - exp(-r * t) * k * pi2
        if self.option_type == 'put':
            y = y - s0 + k
        return y

    def interpolate(self, n_, z):
        z_interpolated = np.zeros((n_, z.shape[1]))
        x = np.linspace(self.df['strike'].min(), self.df['strike'].max(), n_)
        for j in range(len(self.time)):
            cs = PchipInterpolator(self.df['strike'].values, z[:, j])
            z_interpolated[:, j] = cs(x)
        new_time = np.linspace(min(self.time), max(self.time), n_)
        z_interpolated_ = np.zeros((n_, n_))
        for i in range(len(x)):
            cs1 = PchipInterpolator(self.time, z_interpolated[i, :])
            z_interpolated_[i, :] = cs1(new_time)
        return z_interpolated_, new_time, x

    def plot_surface(self, v, z):
        z_, y_, x_ = self.interpolate(100, z)
        fig = go.Figure()
        fig.add_surface(x=y_, y=x_, z=z_, colorbar={'x': 1.2, 'thickness': 10})
        x, y = self.df['strike'].values, self.time

        for i in range(len(self.time)):
            fig.add_trace(
                go.Scatter3d(y=x, x=[y[i]] * len(x), z=v[:, i],
                            mode='markers', marker=dict(size=3), name=self.date[i]
                            )
             )
        fig.update_layout(title='Surface - implied volatility from Heston, points - volatility from Black-Scholes',
                          autosize=False,
                          width=900, height=900)
        fig.update_layout(scene=dict(
            xaxis_title='Time',
            yaxis_title='Strike',
            zaxis_title='Volatility'),
        )
        fig.show()

    def config_to_file(self, file_name):
        c_df = pd.DataFrame()
        c_df['Option type'] = [self.option_type]
        c_df['Price type'] = [self.price_type]
        c_df['Asset price'] = [self.S]
        c_df['r'] = [0.]
        c_df['Today'] = [self.today]
        self.df_params['Date'] = self.date
        c_df = pd.concat([c_df, self.df_params], ignore_index=True, sort=False)
        c_df.dropna()#inplace=True)
        c_df.to_csv("parametres_" + file_name, index=False)

    def volatility_to_file(self, z, file_name):
        vol_df = pd.DataFrame()
        vol_df['strike'] = self.df['strike']
        vol_df = vol_df.set_index('strike')
        hours = 8
        hours_added = datetime.timedelta(hours=hours)
        for i, t in enumerate(self.date):
            t = pd.to_datetime(t, format='%d%b%y') + hours_added
            vol_df[str(t)] = z[:, i]
        vol_df.to_csv(file_name)

    def count_error(self, real_vol, predicted_vol):
        diff = abs(real_vol - predicted_vol) / real_vol * 100
        max_diff = diff.max()
        min_diff = diff.min()
        diff_ = diff.mean()  # np.sqrt((diff**2).sum())
        return diff_, max_diff, min_diff


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Volatility surface')
    parser.add_argument("t", type=str, choices=['ask_c', 'bid_c', 'mid_c', 'ask_p', 'bid_p', 'mid_p'],
                        help="Surface for ask_c/bid_c/mid_c")
    parser.add_argument("f", type=str, help="Output file name")
    parser.add_argument('-p', nargs='+', help='List of file names', required=True)

    args = parser.parse_args()
    type = args.t
    file_name = args.f
    date = args.p

    vol = Volatility(type, date)
    vol.get_data_from_file()

    v = vol.count_vol()
    # print('\nImplied volatility from BS:\n', v)

    # print('\n', vol.time, vol.df['strike'].values)

    price = vol.count_heston_price()  # подсчет цен опционов по Хестону
    # print('\nOption price from Heston:\n', price)

    vol_ = vol.count_heston_volatility(v, price)
    # print('\nImplied volatility from Heston:\n', vol_)

    d1, d2, d3 = vol.count_error(v, vol_)
    print(" Mean error by L1 norm", d1, "%", '\n', "Max error by L1 norm", d2, "%", '\n', "Min error by L1 norm", d3,
         "%")

    vol.plot_surface(v, vol_)
    vol.config_to_file(file_name)
    vol.volatility_to_file(vol_, file_name)
    vol.volatility_to_file(v, '_' + file_name)
