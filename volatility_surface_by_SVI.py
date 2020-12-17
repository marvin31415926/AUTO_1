import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime
from math import *
import numpy as np
from scipy import stats
from scipy.optimize import minimize, Bounds,NonlinearConstraint
from scipy.interpolate import CubicSpline as spl, PchipInterpolator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import datetime
import warnings
warnings.filterwarnings("ignore")

class Price:
    def __init__(self, type):
        if type == 'ask_c' or type == 'bid_c' or type == 'mid_c': 
            self.type = 'call'
        else: 
            self.type = 'put'

class Volatility:
    
    def __init__(self, type, date):
        self.df = pd.DataFrame()
        self.S = 0
        pr = Price(type)
        self.option_type = pr.type
        self.price_type = type
        self.date = date #['28NOV20','4DEC20', '11DEC20', '18DEC20']#'27NOV20', 
        self.time = []
        # hours = 8
        # hours_added = datetime.timedelta(hours = hours)
        # self.today = pd.to_datetime(day, format='%Y-%m-%d:%H:%M')
        # for day in self.date:
        #     if (self.today > (pd.to_datetime(day, format='%d%b%y') + hours_added)):
        #         print("Date is incorrect")
        #     self.time.append(abs(self.today - (pd.to_datetime(day, format='%d%b%y') + hours_added)).days / 365)
        hours = 8
        df1 = pd.read_csv(self.date[0] + '.csv')
        day = df1['q'].values[0][:-2]

        self.today = pd.to_datetime(day, format='%Y-%m-%d %H:%M')
        hours_added = datetime.timedelta(hours=hours)
        hours_added_for_today = datetime.timedelta(hours=self.today.hour, minutes=self.today.minute)

        for d in self.date:
            if self.today + hours_added_for_today > (pd.to_datetime(d, format='%d%b%y') + hours_added):
                print("Date is incorrect")
            diff = abs(self.today - pd.to_datetime(d, format='%d%b%y')-hours_added)# + hours_added_for_today)
            hours, remainder = divmod(diff.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            self.time.append(diff.days / 365 + hours/(365*24) + minutes/(365*24*60))

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

    def N(self,x):  
        return stats.norm.cdf(x)

    def Vega(self, S_, d1, expiry):
        vega = S_*sqrt(1.*expiry/pi/2.)*exp(-0.5*d1**2)
        return vega

    def C_(self, d1, d2, E, r, expiry):
        if self.option_type == 'call':
            return self.S * self.N(d1) - E * exp(-r * (expiry)) * self.N(d2)
        else:
            return -self.S * self.N(-d1) + E * exp(-r * (expiry)) * self.N(-d2)



    def sigma_from_price(self, C, E, expiry, r, error):
        sigma = 1.
        dv = error+1.
        while(abs(dv) > error):
            d1 = (log(self.S/E) + (r + sigma**2/2.)*expiry)/(sigma*sqrt(expiry))
            d2 = d1 - sigma*sqrt(expiry)
            price = self.C_(d1, d2, E, r, expiry)
            vega = self.Vega(self.S, d1, expiry)
            price_error = price-C
            dv = 1.*price_error / vega
            sigma = sigma - dv
        return sigma

    def count_vol(self):
        vol = pd.DataFrame()
        v = np.zeros((len(self.df), len(self.date)))
        for i in range(len(self.df)):
            for j in range(len(self.date)):
                v[i][j] = self.sigma_from_price(self.df.loc[i, 'P'+str(j)], self.df.loc[i,'strike'], self.time[j], 0.,  0.001)
        return v

    def phi(self, theta, gamma):
        return 1./(gamma*theta)*(1.-(1.-np.exp(-gamma*theta))/(gamma*theta))

    def SSVI(self, x, gamma, sigma, rho, t):
        theta = sigma*sigma*t
        p = self.phi(theta, gamma)
        return 0.5*theta*(1.+rho*p*x+np.sqrt((p*x+rho)*(p*x+rho)+1.-rho*rho))

    def svi_calc(self, x, v):
        gamma, rho = x[0], x[1]
        xx, TT = self.df['k'][:].values, self.time
        n = len(xx)
        n1 = len(self.df)
        n2 = len(self.date)
        localVarianceSSVI = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                sigma = v[i][j]
                localVarianceSSVI[i][j] = self.SSVI(xx[i], gamma, sigma, rho, TT[j])#self.Sself.SVI_LocalVarg
        return localVarianceSSVI, len(xx), len(TT)#self.df.loc[i,'Volatility 1 week']

    def Black_Scholes(self, X, T, delta):
        r = 0
        d1 = (log(self.S/X) + (r + delta**2/2.)*T)/(delta*sqrt(T))
        d2 = d1 - delta*sqrt(T)
        if self.option_type == 'call':
            C = self.S*self.N(d1)-X*exp(-r*T)*self.N(d2)
            return C
        else:
            P = X*exp(-r*T)*self.N(-d2)-self.S*self.N(-d1)
            return P

    def B_S(self, x0, v):
        z, n1, n2 = self.svi_calc(x0, v)
        z = np.nan_to_num(z)
        bs = np.zeros((n1,n2))
        n = n1
        for i in range(n1):
            for j in range(n2):
                bs[i][j] = self.Black_Scholes(self.df.loc[i,'strike'], self.time[j], z[i][j])
        return bs

    def min_func(self, x0, v):
        m_price = self.df.iloc[:, 1:-1:].values
        return ((self.B_S(x0, v)-m_price)**2).sum()

    def minimize_(self, v):
        bounds = Bounds([0, -1.], [1., 1.])
        con = lambda x: abs(x[1])-4*x[0]
        nonlinear_constraint = NonlinearConstraint (con, -np.inf, -1)    
        x0 = [0, -0.99]
        res = minimize(self.min_func, x0, args=v, method='trust-constr', bounds=bounds,
         constraints=nonlinear_constraint,
        options={'xtol': 1e-3, 'disp': False})
        return res.x

    def interpolate(self, gamma, rho, n_, v):
        x0 = [gamma, rho]
        z1, n, n2 = self.svi_calc(x0, v)
        z_interpolated = np.zeros((n_, n2))
        x = np.linspace(self.df['k'].min(), self.df['k'].max(), n_)
        new_strike = self.S*np.exp(x)
        for j in range(len(self.time)):
            cs = PchipInterpolator(self.df['k'].values, z1[:, j])
            z_interpolated[:, j] = cs(x)
        new_time = np.linspace(min(self.time), max(self.time), n_)
        z_interpolated_ = np.zeros((n_,n_))
        for i in range(len(x)):
            cs1 = PchipInterpolator(self.time, z_interpolated[i, :])
            z_interpolated_[i, :] = cs1(new_time)
        return z1, z_interpolated_, new_time, x

    def plot_surface(self, z_interpolated, v, n_, new_time, x_):
        x, y = self.df['k'].values, self.time
        fig = go.Figure()
        z_svi = z_interpolated
        for i in range(len(new_time)):
            z_svi[:,i] = z_interpolated[:,i]/new_time[i]
        z_svi = np.sqrt(z_svi)

        fig.add_surface(x=new_time, y=x_, z=z_svi,  colorbar={ 'x':1.2, 'thickness':10})

        for i in range(len(self.time)):
            fig.add_trace(
                go.Scatter3d(y=x,x=[y[i]]*len(x),z=v[:,i], 
                             mode='markers', marker=dict(size=3), name=self.date[i]
                )
            )
        fig.update_layout(title='Surface - implied volatility from svi, points - volatility from Black-Scholes', autosize=False,
                          width=900, height=900)
        fig.update_layout(scene = dict(
                            xaxis_title='Time',
                            yaxis_title='Log moneyness',
                            zaxis_title='volatility'),
                            )
        fig.show()
    
    def config_to_file(self, gamma, rho, type, file_name):
        c_df = pd.DataFrame()
        c_df['Option type'] = ['call']
        c_df['Price type'] = [type]
        c_df['Asset price'] = [self.S]
        c_df['r'] = [0.]
        c_df['gamma from SVI'] = [gamma]
        c_df['rho from SVI'] = [rho]
        c_df['Today'] = [self.today]
        c_df.to_csv("parametres_" + file_name, index=False)

    def volatility_to_file(self, z, file_name):
        vol_df = pd.DataFrame()
        vol_df['strike'] = self.df['strike']
        vol_df = vol_df.set_index('strike')
        hours = 8
        hours_added = datetime.timedelta(hours = hours)
        for i, t in enumerate(self.time):
            z[:,i] = z[:,i]/t
        z = np.sqrt(z)

        for i, t in enumerate(self.date):
            t = pd.to_datetime(t, format='%d%b%y') + hours_added
            vol_df[str(t)] = z[:, i]
        vol_df.to_csv(file_name)

    def count_error(self, real_vol, predicted_vol):
        for i, t in enumerate(self.time):
            predicted_vol[:, i] = predicted_vol[:, i]/t
        predicted_vol = np.sqrt(predicted_vol)
        diff = abs(real_vol - predicted_vol)/real_vol*100
        max_diff = diff.max()
        min_diff = diff.min()
        diff_ = diff.mean()  # np.sqrt((diff**2).sum())
        return diff_, max_diff, min_diff


def main():
    parser = argparse.ArgumentParser(description='Volatility surface')
    parser.add_argument("t", type=str, choices=['ask_c', 'bid_c', 'mid_c', 'ask_p', 'bid_p', 'mid_p'], help="Surface for ask_c/bid_c/mid_c")
    #parser.add_argument("d", type=str, help="Date, YYYY-mm-dd:hh:mm")
    parser.add_argument("f", type=str, help="Output file name")
    parser.add_argument('-p', nargs='+', help='List of file names', required=True)

    args = parser.parse_args()
    type = args.t
    #day = args.d
    file_name = args.f
    date = args.p
    n_ = 100
    vol = Volatility(type, date)
    vol.get_data_from_file()
    v = vol.count_vol()
    gamma, rho = vol.minimize_(v)
    z, z_interpolated, new_time, x_ = vol.interpolate(gamma, rho, n_, v)
    d1, d2, d3 = vol.count_error(v, z)
    print(" Mean error by L1 norm", d1, "%", '\n', "Max error by L1 norm", d2, "%", '\n', "Min error by L1 norm", d3, "%")
    vol.plot_surface(z_interpolated, v, 100, new_time, x_)
    vol.config_to_file(gamma, rho, type, file_name)
    vol.volatility_to_file(z, file_name)

main()
