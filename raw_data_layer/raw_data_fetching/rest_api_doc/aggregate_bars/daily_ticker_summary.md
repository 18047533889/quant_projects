> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Daily Ticker Summary (OHLC)

**Endpoint:** `GET /v1/open-close/{stocksTicker}/{date}`

**Description:**

Retrieve the opening and closing prices for a specific stock ticker on a given date, along with any pre-market and after-hours trade prices. This endpoint provides essential daily pricing details, enabling users to evaluate performance, conduct historical analysis, and gain insights into trading activity outside regular market sessions.

Use Cases: Daily performance analysis, historical data collection, after-hours insights, portfolio tracking.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `stocksTicker` | string | Yes | Specify a case-sensitive ticker symbol. For example, AAPL represents Apple Inc. |
| `date` | string | Yes | The date of the requested open/close in the format YYYY-MM-DD. |
| `adjusted` | boolean | No | Whether or not the results are adjusted for splits.  By default, results are adjusted. Set this to false to get results that are NOT adjusted for splits.  |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `afterHours` | number | The close price of the ticker symbol in after hours trading. |
| `close` | number | The close price for the symbol in the given time period. |
| `from` | string | The requested date. |
| `high` | number | The highest price for the symbol in the given time period. |
| `low` | number | The lowest price for the symbol in the given time period. |
| `open` | number | The open price for the symbol in the given time period. |
| `otc` | boolean | Whether or not this aggregate is for an OTC ticker. This field will be left off if false. |
| `preMarket` | integer | The open price of the ticker symbol in pre-market trading. |
| `status` | string | The status of this request's response. |
| `symbol` | string | The exchange symbol that this item is traded under. |
| `volume` | number | The trading volume of the symbol in the given time period. |

## Sample Response

```json
{
  "afterHours": 322.1,
  "close": 325.12,
  "from": "2023-01-09",
  "high": 326.2,
  "low": 322.3,
  "open": 324.66,
  "preMarket": 324.5,
  "status": "OK",
  "symbol": "AAPL",
  "volume": 26122646
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

request = client.get_daily_open_close_agg(
    "AAPL",
    "2023-01-09",
    adjusted="true",
)

print(request)
