> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Top Market Movers

**Endpoint:** `GET /v2/snapshot/locale/us/markets/stocks/{direction}`

**Description:**

Retrieve snapshot data highlighting the top 20 gainers or losers in the U.S. stock market. Gainers are stocks with the largest percentage increase since the previous day’s close, and losers are those with the largest percentage decrease. To ensure meaningful insights, only tickers with a minimum trading volume of 10,000 are included. Snapshot data is cleared daily at 3:30 AM EST and begins repopulating as exchanges report new information, typically starting around 4:00 AM EST. By focusing on these market movers, users can quickly identify significant price shifts and monitor evolving market dynamics.

Use Cases: Market movers identification, trading strategies, market sentiment analysis, portfolio adjustments.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `direction` | string | Yes | The direction of the snapshot results to return.  |
| `include_otc` | boolean | No | Include OTC securities in the response. Default is false (don't include OTC securities).  |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `status` | string | The status of this request's response. |
| `tickers` | array[object] | An array of snapshot data for the specified tickers. |
| `tickers[].day` | object | The most recent daily bar for this ticker. |
| `tickers[].fmv` | number | Fair market value is only available on Business plans. It is our proprietary algorithm to generate a real-time, accurate, fair market value of a tradable security. For more information, <a rel="nofollow" target="_blank" href="https://massive.com/contact">contact us</a>. |
| `tickers[].lastQuote` | object | The most recent quote for this ticker.  This is only returned if your current plan includes quotes. |
| `tickers[].lastTrade` | object | The most recent trade for this ticker.  This is only returned if your current plan includes trades. |
| `tickers[].min` | object | The most recent minute bar for this ticker. |
| `tickers[].prevDay` | object | The previous day's bar for this ticker. |
| `tickers[].ticker` | string | The exchange symbol that this item is traded under. |
| `tickers[].todaysChange` | number | The value of the change from the previous day. |
| `tickers[].todaysChangePerc` | number | The percentage change since the previous day. |
| `tickers[].updated` | integer | The last updated timestamp. |

## Sample Response

```json
{
  "status": "OK",
  "tickers": [
    {
      "day": {
        "c": 14.2284,
        "dv": "133963.0",
        "h": 15.09,
        "l": 14.2,
        "o": 14.33,
        "v": 133963,
        "vw": 14.5311
      },
      "lastQuote": {
        "P": 14.44,
        "S": 11,
        "p": 14.2,
        "s": 25,
        "t": 1605195929997325600
      },
      "lastTrade": {
        "c": [
          63
        ],
        "ds": "536.0",
        "i": "79372124707124",
        "p": 14.2284,
        "s": 536,
        "t": 1605195848258266000,
        "x": 4
      },
      "min": {
        "av": 133963,
        "c": 14.2284,
        "dav": "133963.0",
        "dv": "6108.0",
        "h": 14.325,
        "l": 14.2,
        "n": 5,
        "o": 14.28,
        "t": 1684428600000,
        "v": 6108,
        "vw": 14.2426
      },
      "prevDay": {
        "c": 0.73,
        "h": 0.799,
        "l": 0.73,
        "o": 0.75,
        "v": 1568097,
        "vw": 0.7721
      },
      "ticker": "PDS",
      "todaysChange": 13.498,
      "todaysChangePerc": 1849.096,
      "updated": 1605195848258266000
    }
  ]
}
```

from massive import RESTClient
from massive.rest.models import (
    TickerSnapshot,
)

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

tickers = client.get_snapshot_direction(
	"stocks",
	direction="gainers",
	)

#print(tickers)

# print ticker with % change
for item in tickers:
    # verify this is a TickerSnapshot
    if isinstance(item, TickerSnapshot):
        # verify this is a float
        if isinstance(item.todays_change_percent, float):
            print("{:<15}{:.2f} %".format(item.ticker, item.todays_change_percent))
