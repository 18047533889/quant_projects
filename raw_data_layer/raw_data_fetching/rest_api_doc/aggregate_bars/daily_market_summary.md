> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Daily Market Summary (OHLC)

**Endpoint:** `GET /v2/aggs/grouped/locale/us/market/stocks/{date}`

**Description:**

Retrieve daily OHLC (open, high, low, close), volume, and volume-weighted average price (VWAP) data for all U.S. stocks on a specified trading date. This endpoint returns comprehensive market coverage in a single request, enabling wide-scale analysis, bulk data processing, and research into broad market performance.

Use Cases: Market overview, bulk data processing, historical research, and portfolio comparison.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `date` | string | Yes | The beginning date for the aggregate window. |
| `adjusted` | boolean | No | Whether or not the results are adjusted for splits.  By default, results are adjusted. Set this to false to get results that are NOT adjusted for splits.  |
| `include_otc` | boolean | No | Include OTC securities in the response. Default is false (don't include OTC securities).  |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `adjusted` | boolean | Whether or not this response was adjusted for splits. |
| `queryCount` | integer | The number of aggregates (minute or day) used to generate the response. |
| `request_id` | string | A request id assigned by the server. |
| `resultsCount` | integer | The total number of results for this request. |
| `status` | string | The status of this request's response. |
| `results` | array[object] | An array of results containing the requested data. |
| `results[].T` | string | The exchange symbol that this item is traded under. |
| `results[].c` | number | The close price for the symbol in the given time period. |
| `results[].h` | number | The highest price for the symbol in the given time period. |
| `results[].l` | number | The lowest price for the symbol in the given time period. |
| `results[].n` | integer | The number of transactions in the aggregate window. |
| `results[].o` | number | The open price for the symbol in the given time period. |
| `results[].otc` | boolean | Whether or not this aggregate is for an OTC ticker. This field will be left off if false. |
| `results[].t` | integer | The Unix millisecond timestamp for the end of the aggregate window. |
| `results[].v` | number | The trading volume of the symbol in the given time period. |
| `results[].vw` | number | The volume weighted average price. |

## Sample Response

```json
{
  "adjusted": true,
  "queryCount": 3,
  "request_id": {
    "description": "A request id assigned by the server.",
    "type": "string"
  },
  "results": [
    {
      "T": "KIMpL",
      "c": 25.9102,
      "h": 26.25,
      "l": 25.91,
      "n": 74,
      "o": 26.07,
      "t": 1602705600000,
      "v": 4369,
      "vw": 26.0407
    },
    {
      "T": "TANH",
      "c": 23.4,
      "h": 24.763,
      "l": 22.65,
      "n": 1096,
      "o": 24.5,
      "t": 1602705600000,
      "v": 25933.6,
      "vw": 23.493
    },
    {
      "T": "VSAT",
      "c": 34.24,
      "h": 35.47,
      "l": 34.21,
      "n": 4966,
      "o": 34.9,
      "t": 1602705600000,
      "v": 312583,
      "vw": 34.4736
    }
  ],
  "resultsCount": 3,
  "status": "OK"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

grouped = client.get_grouped_daily_aggs(
    "2025-11-03",
    adjusted="true",
)

print(grouped)
