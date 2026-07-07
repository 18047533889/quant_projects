> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Unified Snapshot

**Endpoint:** `GET /v3/snapshot`

**Description:**

Retrieve unified snapshots of market data for multiple asset classes including stocks, options, forex, and cryptocurrencies in a single request. This endpoint consolidates key metrics such as last trade, last quote, open, high, low, close, and volume for a comprehensive view of current market conditions. By aggregating data from various sources into one response, users can efficiently monitor, compare, and act on information spanning multiple markets and asset types.

Use Cases: Cross-market analysis, diversified portfolio monitoring, global market insights, multi-asset trading strategies.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `ticker` | string | No | Search a range of tickers lexicographically. |
| `type` | string | No | Query by the type of asset. |
| `ticker.gte` | string | No | Range by ticker. |
| `ticker.gt` | string | No | Range by ticker. |
| `ticker.lte` | string | No | Range by ticker. |
| `ticker.lt` | string | No | Range by ticker. |
| `ticker.any_of` | string | No | Comma separated list of tickers, up to a maximum of 250. If no tickers are passed then all results will be returned in a paginated manner.  Warning: The maximum number of characters allowed in a URL are subject to your technology stack.  |
| `order` | string | No | Order results based on the `sort` field. |
| `limit` | integer | No | Limit the number of results returned, default is 10 and max is 250. |
| `sort` | string | No | Sort field used for ordering. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `next_url` | string | If present, this value can be used to fetch the next page of data. |
| `request_id` | string | A request id assigned by the server. |
| `results` | array[object] | An array of results containing the requested data. |
| `results[].break_even_price` | number | The price of the underlying asset for the contract to break even. For a call, this value is (strike price + premium paid). For a put, this value is (strike price - premium paid). |
| `results[].details` | object | The details for this contract. |
| `results[].error` | string | The error while looking for this ticker. |
| `results[].fmv` | number | Fair Market Value is only available on Business plans. It is our proprietary algorithm to generate a real-time, accurate, fair market value of a tradable security. For more information, <a rel="nofollow" target="_blank" href="https://massive.com/contact">contact us</a>. |
| `results[].fmv_last_updated` | integer | If Fair Market Value (FMV) is available, this field is the nanosecond timestamp of the last FMV calculation. |
| `results[].greeks` | object | The greeks for this contract. There are certain circumstances where greeks will not be returned, such as options contracts that are deep in the money. See this <a href="https://massive.com/blog/greeks-and-implied-volatility/#testing" alt="link">article</a> for more information. |
| `results[].implied_volatility` | number | The market's forecast for the volatility of the underlying asset, based on this option's current price. |
| `results[].last_minute` | object | The most recent minute aggregate for this stock. |
| `results[].last_quote` | object | The most recent quote for this contract. This is only returned if your current plan includes quotes. |
| `results[].last_trade` | object | The most recent quote for this contract. This is only returned if your current plan includes trades. |
| `results[].last_updated` | integer | The nanosecond timestamp of when this information was updated. |
| `results[].market_status` | string | The market status for the market that trades this ticker. Possible values for stocks, options, crypto, and forex snapshots are open, closed, early_trading, or late_trading. Possible values for indices snapshots are regular_trading, closed, early_trading, and late_trading. |
| `results[].message` | string | The error message while looking for this ticker. |
| `results[].name` | string | The name of this contract. |
| `results[].open_interest` | number | The quantity of this contract held at the end of the last trading day. |
| `results[].session` | object | Comprehensive trading session metrics, detailing price changes, trading volume, and key price points (open, close, high, low) for the asset within the current trading day. Includes specific changes during early, regular, and late trading periods to enable detailed performance analysis and trend tracking. |
| `results[].ticker` | string | The ticker symbol for the asset. |
| `results[].timeframe` | enum: DELAYED, REAL-TIME | The time relevance of the data. |
| `results[].type` | enum: stocks, options, fx, crypto, indices | The asset class for this ticker. |
| `results[].underlying_asset` | object | Information on the underlying stock for this options contract.  The market data returned depends on your current stocks plan. |
| `results[].value` | number | Value of Index. |
| `status` | string | The status of this request's response. |

## Sample Response

```json
{
  "request_id": "abc123",
  "results": [
    {
      "break_even_price": 171.075,
      "details": {
        "contract_type": "call",
        "exercise_style": "american",
        "expiration_date": "2022-10-14",
        "shares_per_contract": 100,
        "strike_price": 5,
        "underlying_ticker": "NCLH"
      },
      "fmv": 0.05,
      "fmv_last_updated": 1636573458757383400,
      "greeks": {
        "delta": 0.5520187372272933,
        "gamma": 0.00706756515659829,
        "theta": -0.018532772783847958,
        "vega": 0.7274811132998142
      },
      "implied_volatility": 0.3048997097864957,
      "last_quote": {
        "ask": 21.25,
        "ask_exchange": 12,
        "ask_size": 110,
        "bid": 20.9,
        "bid_exchange": 10,
        "bid_size": 172,
        "last_updated": 1636573458756383500,
        "midpoint": 21.075,
        "timeframe": "REAL-TIME"
      },
      "last_trade": {
        "conditions": [
          209
        ],
        "exchange": 316,
        "price": 0.05,
        "sip_timestamp": 1675280958783136800,
        "size": 2,
        "timeframe": "REAL-TIME"
      },
      "market_status": "closed",
      "name": "NCLH $5 Call",
      "open_interest": 8921,
      "session": {
        "change": -0.05,
        "change_percent": -1.07,
        "close": 6.65,
        "decimal_volume": "67.0",
        "early_trading_change": -0.01,
        "early_trading_change_percent": -0.03,
        "high": 7.01,
        "late_trading_change": -0.4,
        "late_trading_change_percent": -0.02,
        "low": 5.42,
        "open": 6.7,
        "previous_close": 6.71,
        "regular_trading_change": -0.6,
        "regular_trading_change_percent": -0.5,
        "volume": 67
      },
      "ticker": "O:NCLH221014C00005000",
      "type": "options",
      "underlying_asset": {
        "change_to_break_even": 23.123999999999995,
        "last_updated": 1636573459862384600,
        "price": 147.951,
        "ticker": "AAPL",
        "timeframe": "REAL-TIME"
      }
    },
    {
      "fmv": 0.05,
      "fmv_last_updated": 1636573458757383400,
      "last_minute": {
        "close": 412.05,
        "decimal_volume": "610.0",
        "high": 412.1,
        "low": 412.05,
        "open": 412.1,
        "transactions": 26,
        "volume": 610,
        "vwap": 412.0881
      },
      "last_quote": {
        "ask": 21.25,
        "ask_exchange": 300,
        "ask_size": 110,
        "bid": 20.9,
        "bid_exchange": 323,
        "bid_size": 172,
        "last_updated": 1636573458756383500,
        "timeframe": "REAL-TIME"
      },
      "last_trade": {
        "conditions": [
          209
        ],
        "decimal_size": "2.0",
        "exchange": 316,
        "id": "4064",
        "last_updated": 1675280958783136800,
        "price": 0.05,
        "size": 2,
        "timeframe": "REAL-TIME"
      },
      "market_status": "closed",
      "name": "Apple Inc.",
      "session": {
        "change": -1.05,
        "change_percent": -4.67,
        "close": 21.4,
        "decimal_volume": "37.0",
        "early_trading_change": -0.39,
        "early_trading_change_percent": -0.07,
        "high": 22.49,
        "late_trading_change": 1.2,
        "late_trading_change_percent": 3.92,
        "low": 21.35,
        "open": 22.49,
        "previous_close": 22.45,
        "volume": 37
      },
      "ticker": "AAPL",
      "type": "stocks"
    },
    {
      "error": "NOT_FOUND",
      "message": "Ticker not found.",
      "ticker": "TSLAAPL"
    }
  ],
  "status": "OK"
}
```

from typing import cast, Iterator, Union
from urllib3 import HTTPResponse
from massive import RESTClient
from massive.rest.models import UniversalSnapshot, SnapshotMarketType

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

def print_snapshots(iterator: Union[Iterator[UniversalSnapshot], HTTPResponse]):
    snapshots = [s for s in iterator]

    print(f"count: {len(snapshots)}")

    for item in snapshots:
        print(item)


# it = client.list_universal_snapshots() # all tickers for all assets types in lexicographical order

it = client.list_universal_snapshots(
    ticker_any_of=[
        "AAPL",
        "O:AAPL230519C00055000",
        "DOES_NOT_EXIST",
        "X:1INCHUSD",
        "C:AEDAUD",
    ]
)
print_snapshots(it)

it = client.list_universal_snapshots(type="stocks", ticker_gt="A", ticker_lt="AAPL")
print_snapshots(it)

it = client.list_universal_snapshots(type="stocks", ticker_gte="AAPL", ticker_lte="ABB")
print_snapshots(it)

it = client.list_universal_snapshots(
    type="options",
    ticker_gte="O:AAPL230804C00050000",
    ticker_lte="O:AAPL230804C00070000",
)
print_snapshots(it)
