> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Short Volume

**Endpoint:** `GET /stocks/v1/short-volume`

**Description:**

Retrieve daily aggregated short sale volume data reported to FINRA from off-exchange trading venues and alternative trading systems (ATS) for a specified stock ticker. Unlike short interest, which measures outstanding short positions at specific reporting intervals, short volume captures the daily trading activity of short sales. Monitoring short volume helps users detect immediate market sentiment shifts, analyze trading behavior, and identify trends in short-selling activity that may signal upcoming price movements.

Use Cases: Intraday sentiment analysis, short-sale trend identification, liquidity analysis, trading strategy optimization.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `ticker` | string | No | The primary ticker symbol for the stock. |
| `ticker.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `ticker.gt` | string | No | Filter greater than the value. |
| `ticker.gte` | string | No | Filter greater than or equal to the value. |
| `ticker.lt` | string | No | Filter less than the value. |
| `ticker.lte` | string | No | Filter less than or equal to the value. |
| `date` | string | No | The date of trade activity reported in the format YYYY-MM-DD |
| `date.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `date.gt` | string | No | Filter greater than the value. |
| `date.gte` | string | No | Filter greater than or equal to the value. |
| `date.lt` | string | No | Filter less than the value. |
| `date.lte` | string | No | Filter less than or equal to the value. |
| `short_volume_ratio` | number | No | The percentage of total volume that was sold short. Calculated as (short_volume / total_volume) * 100. Value must be a floating point number. |
| `short_volume_ratio.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. Value must be a floating point number. |
| `short_volume_ratio.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `short_volume_ratio.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `short_volume_ratio.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `short_volume_ratio.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `limit` | integer | No | Limit the maximum number of results returned. Defaults to '10' if not specified. The maximum allowed limit is '50000'. |
| `sort` | string | No | A comma separated list of sort columns. For each column, append '.asc' or '.desc' to specify the sort direction. The sort column defaults to 'ticker' if not specified. The sort order defaults to 'asc' if not specified. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `next_url` | string | If present, this value can be used to fetch the next page. |
| `request_id` | string | A request id assigned by the server. |
| `results` | array[object] | The results for this request. |
| `results[].adf_short_volume` | integer | Short volume reported via the Alternative Display Facility (ADF), excluding exempt volume. |
| `results[].adf_short_volume_exempt` | integer | Short volume reported via ADF that was marked as exempt. |
| `results[].date` | string | The date of trade activity reported in the format YYYY-MM-DD |
| `results[].exempt_volume` | number | Portion of short volume that was marked as exempt from regulation SHO. |
| `results[].nasdaq_carteret_short_volume` | integer | Short volume reported from Nasdaq's Carteret facility, excluding exempt volume. |
| `results[].nasdaq_carteret_short_volume_exempt` | integer | Short volume from Nasdaq Carteret that was marked as exempt. |
| `results[].nasdaq_chicago_short_volume` | integer | Short volume reported from Nasdaq's Chicago facility, excluding exempt volume. |
| `results[].nasdaq_chicago_short_volume_exempt` | integer | Short volume from Nasdaq Chicago that was marked as exempt. |
| `results[].non_exempt_volume` | number | Portion of short volume that was not exempt from regulation SHO (i.e., short_volume - exempt_volume). |
| `results[].nyse_short_volume` | integer | Short volume reported from NYSE facilities, excluding exempt volume. |
| `results[].nyse_short_volume_exempt` | integer | Short volume from NYSE facilities that was marked as exempt. |
| `results[].short_volume` | number | Total number of shares sold short across all venues for the ticker on the given date. |
| `results[].short_volume_ratio` | number | The percentage of total volume that was sold short. Calculated as (short_volume / total_volume) * 100. |
| `results[].ticker` | string | The primary ticker symbol for the stock. |
| `results[].total_volume` | number | Total reported volume across all venues for the ticker on the given date. |
| `status` | enum: OK | The status of this request's response. |

## Sample Response

```json
{
  "count": 1,
  "request_id": 1,
  "results": [
    {
      "adf_short_volume": 0,
      "adf_short_volume_exempt": 0,
      "date": "2025-03-25",
      "exempt_volume": 1,
      "nasdaq_carteret_short_volume": 179943,
      "nasdaq_carteret_short_volume_exempt": 1,
      "nasdaq_chicago_short_volume": 1,
      "nasdaq_chicago_short_volume_exempt": 0,
      "non_exempt_volume": 181218,
      "nyse_short_volume": 1275,
      "nyse_short_volume_exempt": 0,
      "short_volume": 181219,
      "short_volume_ratio": 31.57,
      "ticker": "A",
      "total_volume": 574084
    }
  ],
  "status": "OK"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

items = []
for item in client.list_short_volume(
	limit=10,
	sort="ticker.asc",
	):
    items.append(item)

print(items)
