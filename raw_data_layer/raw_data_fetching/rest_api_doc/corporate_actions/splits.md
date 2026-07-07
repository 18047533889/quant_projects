> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Splits

**Endpoint:** `GET /stocks/v1/splits`

**Description:**

Retrieve historical stock split events, including execution dates and ratio factors, to understand changes in a company’s share structure over time. Also find adjustment factors that can be used to normalize historical prices to today's share basis.

Use Cases: Historical analysis, price adjustments, data consistency, modeling.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `ticker` | string | No | Stock symbol for the company that executed the split |
| `ticker.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `ticker.gt` | string | No | Filter greater than the value. |
| `ticker.gte` | string | No | Filter greater than or equal to the value. |
| `ticker.lt` | string | No | Filter less than the value. |
| `ticker.lte` | string | No | Filter less than or equal to the value. |
| `execution_date` | string | No | Date when the stock split was applied and shares adjusted Value must be formatted 'yyyy-mm-dd'. |
| `execution_date.gt` | string | No | Filter greater than the value. Value must be formatted 'yyyy-mm-dd'. |
| `execution_date.gte` | string | No | Filter greater than or equal to the value. Value must be formatted 'yyyy-mm-dd'. |
| `execution_date.lt` | string | No | Filter less than the value. Value must be formatted 'yyyy-mm-dd'. |
| `execution_date.lte` | string | No | Filter less than or equal to the value. Value must be formatted 'yyyy-mm-dd'. |
| `adjustment_type` | string | No | Classification of the share-change event. Possible values include: forward_split (share count increases), reverse_split (share count decreases), stock_dividend (shares issued as a dividend) |
| `adjustment_type.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `limit` | integer | No | Limit the maximum number of results returned. Defaults to '100' if not specified. The maximum allowed limit is '5000'. |
| `sort` | string | No | A comma separated list of sort columns. For each column, append '.asc' or '.desc' to specify the sort direction. The sort column defaults to 'execution_date' if not specified. The sort order defaults to 'desc' if not specified. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `next_url` | string | If present, this value can be used to fetch the next page. |
| `request_id` | string | A request id assigned by the server. |
| `results` | array[object] | The results for this request. |
| `results[].adjustment_type` | string | Classification of the share-change event. Possible values include: forward_split (share count increases), reverse_split (share count decreases), stock_dividend (shares issued as a dividend) |
| `results[].execution_date` | string | Date when the stock split was applied and shares adjusted |
| `results[].historical_adjustment_factor` | number | Cumulative adjustment factor used to offset split effects on historical prices. To adjust a historical price for splits: for a price on date D, find the first split whose `execution_date` is after date D and multiply the unadjusted price by the `historical_adjustment_factor`. |
| `results[].id` | string | Unique identifier for each stock split event |
| `results[].split_from` | number | Denominator of the split ratio (old shares) |
| `results[].split_to` | number | Numerator of the split ratio (new shares) |
| `results[].ticker` | string | Stock symbol for the company that executed the split |
| `status` | enum: OK | The status of this request's response. |

## Sample Response

```json
{
  "request_id": 1,
  "results": [
    {
      "adjustment_type": "forward_split",
      "execution_date": "2005-02-28",
      "historical_adjustment_factor": 0.017857,
      "id": "E90a77bdf742661741ed7c8fc086415f0457c2816c45899d73aaa88bdc8ff6025",
      "split_from": 1,
      "split_to": 2,
      "ticker": "AAPL"
    }
  ],
  "status": "OK"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

splits = []
for s in client.list_stocks_splits(
    limit="100",
    sort="execution_date.desc",
    ):
    splits.append(s)

print(splits)