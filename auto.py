from Naked.toolshed.shell import execute_js
import os


if __name__ == '__main__':
    path = './DATA'
    try:
        os.mkdir(path)
    except OSError:
        print(f'Creation of the directory {path} is failed')

    asset = 'BTC'

    bitcom_script = 'get_all_bit_prices_mm.js'
    bitcom_dates = ['16DEC20', '17DEC20', '18DEC20', '25DEC20']

    deribit_script = 'get_all_deribit_prices_mm.js'
    deribit_dates = ['16DEC20', '17DEC20', '18DEC20', '25DEC20']

    okex_script = 'get_all_okex_prices_mm.js'
    okex_dates = ['16DEC20', '17DEC20', '18DEC20', '25DEC20']

    for date in bitcom_dates:
        result = execute_js(bitcom_script + ' ' + asset + '-' + date + '>' + './DATA/' + date + '_b' + '.csv')
        print('Bit.com', date, result)

    for date in deribit_dates:
        result = execute_js(deribit_script + ' ' + asset + '-' + date + '>' + './DATA/' + date + '_d' + '.csv')
        print('Deribit', date, result)

    for date in okex_dates:
        result = execute_js(okex_script + ' ' + asset + '-' + date + '>' + './DATA/' + date + '_o' + '.csv')
        print('OKEx', date, result)
        # 000