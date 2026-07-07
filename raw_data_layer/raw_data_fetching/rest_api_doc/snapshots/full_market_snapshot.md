> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Full Market Snapshot

**Endpoint:** `GET /v2/snapshot/locale/us/markets/stocks/tickers`

**Description:**

Retrieve a comprehensive snapshot of the entire U.S. stock market, covering over 10,000+ actively traded tickers in a single response. This endpoint consolidates key information like pricing, volume, and trade activity to provide a full-market-snapshot view, eliminating the need for multiple queries. Snapshot data is cleared daily at 3:30 AM EST and begins to repopulate as exchanges report new data, which can start as early as 4:00 AM EST. By accessing all tickers at once, users can efficiently monitor broad market conditions, perform bulk analyses, and power applications that require complete, current market information.

Use Cases: Market overview, bulk data processing, heat maps/dashboards, automated monitoring.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `tickers` | array | No | A case-sensitive comma separated list of tickers to get snapshots for. For example, AAPL,TSLA,GOOG. Empty string defaults to querying all tickers. |
| `include_otc` | boolean | No | Include OTC securities in the response. Default is false (don't include OTC securities).  |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `count` | integer | The total number of results for this request. |
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
  "count": 1,
  "status": "OK",
  "tickers": [
    {
      "day": {
        "c": 20.506,
        "dv": "37216.0",
        "h": 20.64,
        "l": 20.506,
        "o": 20.64,
        "v": 37216,
        "vw": 20.616
      },
      "lastQuote": {
        "P": 20.6,
        "S": 22,
        "p": 20.5,
        "s": 13,
        "t": 1605192959994246100
      },
      "lastTrade": {
        "c": [
          14,
          41
        ],
        "ds": "2416.0",
        "i": "71675577320245",
        "p": 20.506,
        "s": 2416,
        "t": 1605192894630916600,
        "x": 4
      },
      "min": {
        "av": 37216,
        "c": 20.506,
        "dav": "37216.0",
        "dv": "5000.0",
        "h": 20.506,
        "l": 20.506,
        "n": 1,
        "o": 20.506,
        "t": 1684428600000,
        "v": 5000,
        "vw": 20.5105
      },
      "prevDay": {
        "c": 20.63,
        "h": 21,
        "l": 20.5,
        "o": 20.79,
        "v": 292738,
        "vw": 20.6939
      },
      "ticker": "BCAT",
      "todaysChange": -0.124,
      "todaysChangePerc": -0.601,
      "updated": 1605192894630916600
    }
  ]
}
```

from massive import RESTClient
from massive.rest.models import (
    TickerSnapshot,
    Agg,
)

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

snapshot = client.get_snapshot_all(
	"stocks",
	)

print(snapshot)

# crunch some numbers
for item in snapshot:
    # verify this is an TickerSnapshot
    if isinstance(item, TickerSnapshot):
        # verify this is an Agg
        if isinstance(item.prev_day, Agg):
            # verify this is a float
            if isinstance(item.prev_day.open, float) and isinstance(
                item.prev_day.close, float
            ):
                percent_change = (
                    (item.prev_day.close - item.prev_day.open)
                    / item.prev_day.open
                    * 100
                )
                print(
                    "{:<15}{:<15}{:<15}{:.2f} %".format(
                        item.ticker,
                        item.prev_day.open,
                        item.prev_day.close,
                        percent_change,
                    )
                )
