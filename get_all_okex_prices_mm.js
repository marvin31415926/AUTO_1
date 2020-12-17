var fetch = require('node-fetch');
const { argv } = require('process');
function timeConverter(timestamp){
    var a = new Date(timestamp);//.toLocaleString( "en-US", { timeZone: "America/New_York"});
    var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    var year = a.getFullYear();
    var month = a.getUTCMonth() + 1;
    var date = a.getUTCDate();
    var hour = a.getUTCHours();
    var min = a.getUTCMinutes();
    var sec = a.getUTCSeconds();

    if (month < 10)
        month = '0' + month;
    else
    if (date < 10)
        date = '0' + date;

    if (hour < 10)
        hour = '0' + hour;

    if (min < 10)
        min = '0' + min;


    if (sec < 10)
        sec = '0' + sec;

    var time = year + '-' + month + '-' + date + ' '+ hour + ':' + min + ':' + sec;
    //console.log(time);
    return time;
}
from_currency_get_maturity = (currency) =>
{
    let arr = currency.split('-');
    let a = new Date(arr[1]+'Z');

    var year = a.getFullYear();
    var month = a.getUTCMonth() + 1;
    var date = a.getUTCDate();
    var hour = a.getUTCHours();
    var min = a.getUTCMinutes();
    var sec = a.getUTCSeconds();

    if (month < 10)
        month = '0' + month;
    else
    if (date < 10)
        date = '0' + date;

    if (hour < 10)
        hour = '0' + (hour + 8) ;

    if (min < 10)
        min = '0' + min;


    if (sec < 10)
        sec = '0' + sec;

    var time = year + '-' + month + '-' + date + ' '+ hour + ':' + min + ':' + sec;
    return time;

}

date_now_utc = () => {
    return new Date().toJSON().slice(0,19).toString().replace('T', ' ');
    //return new Date().toJSON();
}

get_index_price = async (tok_name) => {
    //https://www.okex.com/api/index/v3/BTC-USD/constituents
    let response = await fetch('https://www.okex.com/api/index/v3/' + tok_name + '-USD/constituents').
    then(res => res.json())
    .catch(err => {
        console.log(date_now_utc());
        console.log(err.message + '\n');
    })
    ;
    if (response.code != 0)
        {
            console.log(response.error_message);
            console.log("Returned code " + response.code);
            return 1;
        }
    return response.data.last;
}

get_instrument = async (iname, strike, count, time_response, underlying_price) => {

    //https://www.okex.com/api/option/v3/instruments/ETH-USD/summary/ETH-USD-201212-600-C
    let maturityforurl = from_currency_get_maturity(iname).slice(2).split(' ')[0].replace(/-/g, '');
    let call_url = 'https://www.okex.com/api/option/v3/instruments/' + iname.split('-')[0] + '-USD/summary/' + iname.split('-')[0] + '-USD-' + maturityforurl+'-' + strike + '-C';
    //let call_url = 'https://betaapi.bitexch.dev/v1/tickers?instrument_id='+ iname + '-' + strike + '-C';
    //let put_url = 'https://betaapi.bitexch.dev/v1/tickers?instrument_id='+ iname + '-' + strike + '-P';
    let put_url = 'https://www.okex.com/api/option/v3/instruments/' + iname.split('-')[0] + '-USD/summary/' + iname.split('-')[0] + '-USD-' + maturityforurl+'-' + strike + '-P';

    //https://www.okex.com/api/option/v3/instruments/BTC-USD/summary

    //console.log(call_url);
    
    let response_call = await fetch(call_url)
    .then( res => res.json());
    let response_put = await fetch(put_url)
    .then (res => res.json())
    .catch(err => {
        console.log(date_now_utc());
        console.log(err.message + '\n');
    })
    ;

    let bid_c = "";
    let ask_c = "";
    let bid_p = "";
    let ask_p = "";
    let maturity = ""; // общее поле
    //let time_response = ""; // общее поле
    //let underlying_price = ""; // общее поле
    let delta = "";
    //console.log(response_call);
    if (!response_call.error_message)
    {
        bid_c = response_call.best_bid;
        ask_c = response_call.best_ask;
        //underlying_price = response_call.data.underlying_price;
        delta = response_call.delta;
        //time_response = timeConverter(response_call.data.time);

    }
    if (!response_put.error_message)
    {
        bid_p = response_put.best_bid;
        ask_p = response_put.best_ask;
        //underlying_price = response_put.data.underlying_price;
        delta = response_put.delta;
        //time_response = timeConverter(response_put.data.time);
    }

    let str = "";
    maturity = from_currency_get_maturity(iname);
    //console.log("delta: " + delta);
    if (delta != "")
    {
        let print_bid_c = (Math.round(+bid_c * underlying_price * 10000) / 10000 == 0 ? "" : Math.round(+bid_c * underlying_price * 10000) / 10000);
        let print_ask_c = (Math.round(+ask_c * underlying_price * 10000) / 10000 == 0 ? "" : Math.round(+ask_c * underlying_price * 10000) / 10000);
        let print_bid_p = (Math.round(+bid_p * underlying_price * 10000) / 10000 == 0 ? "" : Math.round(+bid_p * underlying_price * 10000) / 10000);
        let print_ask_p = (Math.round(+ask_p * underlying_price * 10000) / 10000 == 0 ? "" : Math.round(+ask_p * underlying_price * 10000) / 10000);
        str = count + ',' + strike
            + ',' + print_bid_c
            + ',' + print_ask_c
            + ',' + print_bid_p
            + ',' + print_ask_p
            + ',' + maturity
            + ',' + time_response
            + ',' + "0" + "," + "0" + ',' +  Math.round(underlying_price * 10000) / 10000
            + ',' + "okex"
            +'\n';


    }
    return str;
}


create_table = async (iname) => {
    var result = ",k,bid_c,ask_c,bid_p,ask_p,e,q,qty_p,qty_c,s0,name\n";

    let underlying_price = await get_index_price(iname.split('-')[0]);
    if (underlying_price == 1)
        return;
    let time = date_now_utc();
    let k = 0;

    let all_url = 'https://www.okex.com/api/option/v3/instruments/'+ iname.split('-')[0] + '-USD/summary';

    let strikes = [];
    let response_all = await fetch(all_url).
    then(res => res.json())
    .catch(err => {
        console.log(date_now_utc());
        console.log(err.message + '\n');
    })
    ;
    let maturityforurl = from_currency_get_maturity(iname).slice(2).split(' ')[0].replace(/-/g, '');

    for (let i = 0; i < response_all.length; ++i)
    {
        if (response_all[i].instrument_id.indexOf(iname.split('-')[0] + '-USD-' + maturityforurl+'-') != -1)
        {
            if (strikes.findIndex(el => el == response_all[i].instrument_id.split('-')[3]) == -1 )
            {
                strikes.push(response_all[i].instrument_id.split('-')[3]);
            }
        }
    }
    //console.log(strikes);
    //return;
    strikes.sort();
    for (let i = 0; i < strikes.length; ++i)
    {
        let str = await get_instrument(iname, strikes[i], k, time, underlying_price);
        if (str == "")
         continue;
        result+=str;
        ++k;
    }
    if (result == ",k,bid_c,ask_c,bid_p,ask_p,e,q,qty_p,qty_c,s0,name\n")
      {
        console.log("There are no options with these strikes");
        console.log("Returned code 1");
        return;
      }
    console.log(result);
    return;
}
get_deribit_prices_mm = () => {

    if (process.argv.length < 3)
    {

        // console.log("Not enough arguments. Minimus is 3")
        console.log("usage: node get_bit_prices_mm.js <u_token-DATE> <strike1>, ...");
        return;
    }
    // if (isNaN(+process.argv[2]))
    // {
    //     console.log("Incorrect period parameter or missed");
    //     return;
    // }
    let period = 0;
    let iname = process.argv[2];
    let k = 4;

    const exec = () => {
        let timer;
        create_table(iname);
    if (period != 0)
        timer = setTimeout(exec, period * 1000);
    }
    exec();
}

get_deribit_prices_mm();
//from_currency_get_maturity("BTC-12DEC20");
//console.log(from_currency_get_maturity('BTC-12DEC20').slice(2).split(' ')[0].replace(/-/g, ''));
