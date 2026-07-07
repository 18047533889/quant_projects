> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Simple Moving Average (SMA)

**Endpoint:** `GET /v1/indicators/sma/{stockTicker}`

**Description:**

Retrieve the Simple Moving Average (SMA) for a specified ticker over a defined time range. The SMA calculates the average price across a set number of periods, smoothing price fluctuations to reveal underlying trends and potential signals.

Use Cases: Trend analysis, trading signal generation (e.g., SMA crossovers), identifying support/resistance, and refining entry/exit timing.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `stockTicker` | string | Yes | Specify a case-sensitive ticker symbol for which to get simple moving average (SMA) data. For example, AAPL represents Apple Inc. |
| `timestamp` | string | No | Query by timestamp. Either a date with the format YYYY-MM-DD or a millisecond timestamp. |
| `timespan` | string | No | The size of the aggregate time window. |
| `adjusted` | boolean | No | Whether or not the aggregates used to calculate the simple moving average are adjusted for splits. By default, aggregates are adjusted. Set this to false to get results that are NOT adjusted for splits. |
| `window` | integer | No | The window size used to calculate the simple moving average (SMA). i.e. a window size of 10 with daily aggregates would result in a 10 day moving average. |
| `series_type` | string | No | The price in the aggregate which will be used to calculate the simple moving average. i.e. 'close' will result in using close prices to  calculate the simple moving average (SMA). |
| `expand_underlying` | boolean | No | Whether or not to include the aggregates used to calculate this indicator in the response. |
| `order` | string | No | The order in which to return the results, ordered by timestamp. |
| `limit` | integer | No | Limit the number of results returned, default is 10 and max is 5000 |
| `timestamp.gte` | string | No | Range by timestamp. |
| `timestamp.gt` | string | No | Range by timestamp. |
| `timestamp.lte` | string | No | Range by timestamp. |
| `timestamp.lt` | string | No | Range by timestamp. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `next_url` | string | If present, this value can be used to fetch the next page of data. |
| `request_id` | string | A request id assigned by the server. |
| `results` | object | The results of the SMA indicator calculation. |
| `results.underlying` | object | The underlying aggregates used. |
| `results.values` | array[object] | Timestamp or indicator value. |
| `status` | string | The status of this request's response. |

## Sample Response

```json
{
  "next_url": "https://api.massive.com/v1/indicators/sma/AAPL?cursor=YWN0aXZlPXRydWUmZGF0ZT0yMDIxLTA0LTI1JmxpbWl0PTEmb3JkZXI9YXNjJnBhZ2VfbWFya2VyPUElN0M5YWRjMjY0ZTgyM2E1ZjBiOGUyNDc5YmZiOGE1YmYwNDVkYzU0YjgwMDcyMWE2YmI1ZjBjMjQwMjU4MjFmNGZiJnNvcnQ9dGlja2Vy",
  "request_id": "a47d1beb8c11b6ae897ab76cdbbf35a3",
  "results": {
    "underlying": {
      "aggregates": [
        {
          "c": 75.0875,
          "h": 75.15,
          "l": 73.7975,
          "n": 1,
          "o": 74.06,
          "t": 1577941200000,
          "v": 135647456,
          "vw": 74.6099
        },
        {
          "c": 74.3575,
          "h": 75.145,
          "l": 74.125,
          "n": 1,
          "o": 74.2875,
          "t": 1578027600000,
          "v": 146535512,
          "vw": 74.7026
        }
      ],
      "url": "https://api.massive.com/v2/aggs/ticker/AAPL/range/1/day/2003-01-01/2022-07-25"
    },
    "values": [
      {
        "timestamp": 1517562000016,
        "value": 140.139
      }
    ]
  },
  "status": "OK"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

sma = client.get_sma(
    ticker="AAPL",
	timespan="day",
	adjusted="true",
	window="50",
	series_type="close",
	order="desc",
	limit="10",
)

print(sma)
