> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Previous Day Bar (OHLC)

**Endpoint:** `GET /v2/aggs/ticker/{stocksTicker}/prev`

**Description:**

Retrieve the previous trading day's open, high, low, and close (OHLC) data for a specified stock ticker. This endpoint provides key pricing metrics, including volume, to help users assess recent performance and inform trading strategies.

Use Cases: Baseline comparison, technical analysis, market research, and daily reporting.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `stocksTicker` | string | Yes | Specify a case-sensitive ticker symbol. For example, AAPL represents Apple Inc. |
| `adjusted` | boolean | No | Whether or not the results are adjusted for splits.  By default, results are adjusted. Set this to false to get results that are NOT adjusted for splits.  |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `ticker` | string | The exchange symbol that this item is traded under. |
| `adjusted` | boolean | Whether or not this response was adjusted for splits. |
| `queryCount` | integer | The number of aggregates (minute or day) used to generate the response. |
| `request_id` | string | A request id assigned by the server. |
| `resultsCount` | integer | The total number of results for this request. |
| `status` | string | The status of this request's response. |
| `results` | array[object] | An array of results containing the requested data. |
| `results[].c` | number | The close price for the symbol in the given time period. |
| `results[].h` | number | The highest price for the symbol in the given time period. |
| `results[].l` | number | The lowest price for the symbol in the given time period. |
| `results[].n` | integer | The number of transactions in the aggregate window. |
| `results[].o` | number | The open price for the symbol in the given time period. |
| `results[].t` | integer | The Unix millisecond timestamp for the start of the aggregate window. |
| `results[].v` | number | The trading volume of the symbol in the given time period. |
| `results[].vw` | number | The volume weighted average price. |

## Sample Response

```json
{
  "adjusted": true,
  "queryCount": 1,
  "request_id": "6a7e466379af0a71039d60cc78e72282",
  "results": [
    {
      "T": "AAPL",
      "c": 115.97,
      "h": 117.59,
      "l": 114.13,
      "o": 115.55,
      "t": 1605042000000,
      "v": 131704427,
      "vw": 116.3058
    }
  ],
  "resultsCount": 1,
  "status": "OK",
  "ticker": "AAPL"
}
```



from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

aggs = client.get_previous_close_agg(
    "AAPL",
    adjusted="true",
)

print(aggs)
