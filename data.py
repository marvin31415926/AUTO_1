import pandas as pd


class Price:
    def __init__(self, type):
        if type == 'ask_c' or type == 'bid_c' or type == 'mid_c':
            self.type = 'call'
        else:
            self.type = 'put'


class OptionType:
    call = 'call'
    put = 'put'


class OptionTable:
    def __init__(self):
        self.df
        self.request_date
        self.spot_price
        self.expiration_times
        self.strikes
        self.k


        self.df = pd.DataFrame()
        self.S = 0
        pr = Price(type)
        self.option_type = pr.type
        self.price_type = type
        self.date = date
        self.time = []


class DataReader:
    def __init__(self, type, date):
        self.df = pd.DataFrame()
        self.S = 0
        pr = Price(type)
        self.option_type = pr.type
        self.price_type = type
        self.date = date
        self.time = []
        self.df_params = pd.DataFrame(columns=['v0', 'vbar', 'a', 'vvol', 'rho'])

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