const WebSocket = require('ws');
const fs = require('fs');
const { SSL_OP_SSLEAY_080_CLIENT_DH_BUG } = require('constants');
const { argv } = require('process');
const { timingSafeEqual } = require('crypto');
const { POINT_CONVERSION_HYBRID } = require('constants');


date_now_utc = () => {
    //return new Date().toJSON().slice(0,19).toString();
    return new Date().toJSON();
}

date_now_new_york = () => {
    const timezone = "America/New_York";
    const usaTime = new Date().toLocaleString( "en-US", { timeZone: timezone});
    let arr = usaTime.split('/');
    let year_and_hours = arr[2].split(','); //year_and_hours[0] - year
    let hours = year_and_hours[1].split(":");
    if (hours[0] < 10)
        hours[0] = '0' + hours[0].slice(1);
    var time = year_and_hours[0] + '-' + arr[0] + '-' + arr[1] + ''+ hours[0] + ':' + hours[1] + ':' + hours[2].slice(0,2);
    return time;
}
function timeConverter(timestamp){
    var a = new Date(timestamp);//.toLocaleString( "en-US", { timeZone: "America/New_York"});
    var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    var year = a.getFullYear();
    //var month = months[a.getMonth()];
    //var day = a.getDay();
	//a = new Date(a);
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

function time_to_timestamp(date) { 

    var a = new Date(date).toLocaleString( "en-US", { timeZone: "America/New_York"});
    //var a = new Date(date);
    //console.log(+new Date(a));
    return +new Date(a);
}

get_book_summary_by_instrument_with_params = (currency, lst) => {

    var msg = {
        "jsonrpc" : "2.0",
        "id" : 7617,
        "method" : "public/get_instruments",
        "params" : {
            "currency" : currency.split('-')[0],
            "kind" : "option",
            "expired" : false
        }
    };
    
    var msg_underline = {
        "jsonrpc": "2.0",
        "method": "public/get_index_price",
        "id": 42,
        "params": {
        "index_name": currency.split('-')[0].toLowerCase() + "_usd"}
    };

    var ws = new WebSocket('wss://www.deribit.com/ws/api/v2');

    let instrument_name; // имя опциона
    let mark_price; // рыночная
    let bid_price; // бид
    let ask_price; // аск
    let underline_price; // токен

    //let exp_timestamp = time_to_timestamp(expiration_time); // timestamp expiration
	//console.log(exp_timestamp);
    //массивы которые будут заполнены в ходе запросов, из которых будет напечатан результат
    let optionlst = [];
    let countlst = [];
    let bid_c = [];
    let bid_p = [];
    let ask_c = [];
    let ask_p = [];
    let mark_p = [];
    let mark_c = [];
    let strikes = [];
    let exp_time;
    ws.onmessage = (e) => {
    
        let obj =  JSON.parse(e.data); // объект

        if (obj.id === 7617) // создание Set из опционов, которые которые удовлетворяют strike и expiration_time
        {
            if (obj.error != null) {
                console.log("Error: " + obj.error.message);
                console.log("Reason: " + obj.error.data.reason + " with param: " + obj.error.data.param);
                return 0;
            }
            //console.log(exp_timestamp);
            for (let i = 0; i < obj.result.length; ++i) {

				/*console.log(lst.findIndex( elem => elem == obj.result[i].strike));
				console.log("elem: " + Math.ceil(obj.result[i].strike.toString()));
				console.log("arr: " + lst);*/
				//console.log(exp_timestamp);
				//console.log(obj.result[i]);
                //console.log(obj.result[i].expiration_timestamp);
                //27NOV20 ==
                if (obj.result[i].instrument_name.split('-')[1] == currency.split('-')[1]/* && lst.findIndex( elem => elem == obj.result[i].strike) != -1*/)
				{
                    optionlst.push(obj.result[i].instrument_name.slice(0,-1));
                    exp_time = obj.result[i].expiration_timestamp;
				}
            }
            
            let set = new Set(optionlst);

            optionlst = [...set].sort( (a, b) => {
                if (+a.split('-')[2] > +b.split('-')[2])
                    return 1;
                else if (+a.split('-')[2] < +b.split('-')[2])
                    return -1;
                else
                    return 0;
            });
            //console.log(optionlst);
            for (let i = 0; i < optionlst.length; ++i)
            {
                bid_c.push("");
                bid_p.push("");
                ask_c.push("");
                ask_p.push("");
                mark_c.push("");
                mark_p.push("");
                strikes.push("");

            }
            for (let i = 0; i < optionlst.length; ++i)
            {
                let arr = optionlst[i].split('-');
                strikes[i] = arr[arr.length - 2];
                //console.log(typeof(arr[arr.length - 2]));
            }
            var msgall = 
            {
                "jsonrpc" : "2.0",
                "id" : 9344,
                "method" : "public/get_book_summary_by_currency",
                "params" : {
                  "currency" : currency.split('-')[0],
                  "kind" : "option"
                }
            };

            if (!optionlst.length)
            {
                console.log("There are no options with these strikes");
                console.log("Returned code 1");
                ws.close();
            }
            else
                ws.send(JSON.stringify(msgall));

        } // end id 7617

        else if (obj.id === 9344) // информация по всем токенам, чтобы получить цены
        {

            if (obj.error != null) {
                console.log("Error: " + obj.error.message);
                console.log("Reason: " + obj.error.data.reason + " with param: " + obj.error.data.param);
                ws.close();
            }
            else
            {
                for (let i = 0; i < obj.result.length; ++i) {

                    for (let j = 0; j < optionlst.length; ++j)
                    {
                        if (optionlst[j] + 'C' === obj.result[i].instrument_name)
                        {
                            //заполняем бид и аск для call
                            bid_c[j] = (obj.result[i].bid_price === null) ? "" : obj.result[i].bid_price;
                            ask_c[j] = (obj.result[i].ask_price === null) ? "" : obj.result[i].ask_price;
                            mark_c[j] = (obj.result[i].mark_price === null) ? "" : obj.result[i].mark_price;
                        }
                        else if (optionlst[j] + 'P' === obj.result[i].instrument_name)
                        {
                            //заполняем бид и аск для put
                            bid_p[j] = (obj.result[i].bid_price === null) ? "" : obj.result[i].bid_price;
                            ask_p[j] = (obj.result[i].ask_price === null) ? "" : obj.result[i].ask_price;
                            mark_p[j] = (obj.result[i].mark_price === null) ? "" : obj.result[i].mark_price;
                        }
                        
                    }

                } // end for throw obj.result

                var result = ",k,bid_c,ask_c,mark_c,bid_p,ask_p,mark_p,e,q,qty_p,qty_c,s0,name\n";
				
				//debug
				//console.log(optionlst);
				/*console.log("ask_p");
				console.log(ask_p);
				
				console.log("bid_p");
				console.log(bid_p);
				
				console.log("bid_c");
				console.log(bid_c);
				
				console.log("ask_c");
				console.log(ask_c);
				*/
				
				let k = 0;
				//console.log(strikes);
				/*console.log(strikes);
				console.log(lst);*/
				// == "0" ? "" : (bid_c[j] * underline_price).toString()
                for (let j = 0; j < optionlst.length; ++j) {
					
					//if (lst.findIndex(elem => elem == +strikes[j]) != -1)
					{
						bid_c[j] = bid_c[j] * underline_price == 0 ? "" : Math.round(bid_c[j] * underline_price * 10000) / 10000;
						ask_c[j] = ask_c[j] * underline_price == 0 ? "" : Math.round(ask_c[j] * underline_price * 10000) / 10000;
						bid_p[j] = bid_p[j] * underline_price == 0 ? "" : Math.round(bid_p[j] * underline_price * 10000) / 10000;
                        ask_p[j] = ask_p[j] * underline_price === 0 ? "" : Math.round(ask_p[j] * underline_price * 10000) / 10000;
                        
                        mark_c[j] = mark_c[j] * underline_price == 0 ? "" : Math.round(mark_c[j] * underline_price * 10000) / 10000;
                        mark_p[j] = mark_p[j] * underline_price === 0 ? "" : Math.round(mark_p[j] * underline_price * 10000) / 10000;
						let str = k.toString() + ',' + strikes[j].toString()
						+ ',' + bid_c[j]
                        + ',' + ask_c[j]
                        + ',' + mark_c[j]
						+ ',' + bid_p[j]
                        + ',' + ask_p[j]
                        + ',' + mark_p[j]
						+ ',' +  timeConverter(exp_time).replace('T', ' ')
						+ ',' + date_now_utc().replace('T', ' ')
                        + ',' + "0" + "," + "0" + ',' +  underline_price.toString()
                        + ',' + "deribit"
                        +'\n';
						result += str;
						++k;
					}
                }

                ws.close();
                console.log(result);
            }

        } // end id 9344

        else if (obj.id === 42) // если пришел запрос о цене underlying token
        {
            if (obj.error != null)
            {
                console.log("Error: " + obj.error.message);
                console.log("Reason: " + obj.error.data.reason + " with param: " + obj.error.data.param);
                console.log("Returned code -32602");
                ws.close();
            }
            else
            {
                underline_price = obj.result.index_price;
                ws.send(JSON.stringify(msg));
            }

        } // end id 42
    };
    ws.onopen = function () {
        ws.send(JSON.stringify(msg_underline));
    };
} // end get book


get_deribit_prices_mm = () => {

    if (process.argv.length < 3)
    {

        console.log("Not enough arguments. Minimus is 3")
        console.log("usage: node get_deribit_prices_mm.js <u_token-DATE> <strike1>, ...");
        return;
    }

    let currency = process.argv[2];
    /*let k = 3;

    let lst = []; // лист страйков
    while(k < process.argv.length)
    {
        lst.push(argv[k]);
        ++k;
    }*/

    //debug without params
    /*let currency = "BTC";
    let expire_time = "2021-08-24T18:00:00";
    //let expire_time = 1632470400000;
    lst = [7000, 10000];
    */

    const exec_book = () => {

        get_book_summary_by_instrument_with_params(currency);
    }
    exec_book();
}

/*debug time*/
//console.log(date_now_utc());
//console.log(date_now_new_york());
//console.log(time_to_timestamp('2021-06-25T04:00:00'));
//console.log(timeConverter(1624608000000));

get_deribit_prices_mm();