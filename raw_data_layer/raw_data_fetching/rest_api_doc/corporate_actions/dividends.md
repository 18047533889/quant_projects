> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Dividends

**Endpoint:** `GET /stocks/v1/dividends`

**Description:**

Retrieve a historical record of cash dividend distributions for a given ticker, including declaration, ex-dividend, record, and pay dates, as well as payout amounts and adjustment factors for normalizing historical data to offset the effects of dividends. This endpoint consolidates key dividend information, enabling users to account for dividend income in returns, develop dividend-focused strategies, and support tax reporting needs.

Use Cases: Income analysis, total return calculations, dividend strategies, tax planning.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `ticker` | string | No | Stock symbol for the company issuing the dividend |
| `ticker.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `ticker.gt` | string | No | Filter greater than the value. |
| `ticker.gte` | string | No | Filter greater than or equal to the value. |
| `ticker.lt` | string | No | Filter less than the value. |
| `ticker.lte` | string | No | Filter less than or equal to the value. |
| `ex_dividend_date` | string | No | Date when the stock begins trading without the dividend value Value must be formatted 'yyyy-mm-dd'. |
| `ex_dividend_date.gt` | string | No | Filter greater than the value. Value must be formatted 'yyyy-mm-dd'. |
| `ex_dividend_date.gte` | string | No | Filter greater than or equal to the value. Value must be formatted 'yyyy-mm-dd'. |
| `ex_dividend_date.lt` | string | No | Filter less than the value. Value must be formatted 'yyyy-mm-dd'. |
| `ex_dividend_date.lte` | string | No | Filter less than or equal to the value. Value must be formatted 'yyyy-mm-dd'. |
| `frequency` | integer | No | How many times per year this dividend is expected to occur. A value of 0 means the distribution is non-recurring or irregular (e.g., special, supplemental, or a one-off dividend). Other possible values include 1 (annual), 2 (semi-annual), 3 (trimester), 4 (quarterly), 12 (monthly), 24 (bi-monthly), 52 (weekly), 104 (bi-weekly), and 365 (daily) depending on the issuer's declared or inferred payout cadence. Value must be an integer. |
| `frequency.gt` | integer | No | Filter greater than the value. Value must be an integer. |
| `frequency.gte` | integer | No | Filter greater than or equal to the value. Value must be an integer. |
| `frequency.lt` | integer | No | Filter less than the value. Value must be an integer. |
| `frequency.lte` | integer | No | Filter less than or equal to the value. Value must be an integer. |
| `distribution_type` | string | No | Classification describing the nature of this dividend's recurrence pattern: recurring (paid on a regular schedule), special (one-time or commemorative), supplemental (extra beyond the regular schedule), irregular (unpredictable or non-recurring), unknown (cannot be classified from available data) |
| `distribution_type.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `limit` | integer | No | Limit the maximum number of results returned. Defaults to '100' if not specified. The maximum allowed limit is '5000'. |
| `sort` | string | No | A comma separated list of sort columns. For each column, append '.asc' or '.desc' to specify the sort direction. The sort column defaults to 'ticker' if not specified. The sort order defaults to 'asc' if not specified. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `next_url` | string | If present, this value can be used to fetch the next page. |
| `request_id` | string | A request id assigned by the server. |
| `results` | array[object] | The results for this request. |
| `results[].cash_amount` | number | Original dividend amount per share in the specified currency |
| `results[].currency` | string | Currency code for the dividend payment (e.g., USD, CAD) |
| `results[].declaration_date` | string | Date when the company officially announced the dividend |
| `results[].distribution_type` | string | Classification describing the nature of this dividend's recurrence pattern: recurring (paid on a regular schedule), special (one-time or commemorative), supplemental (extra beyond the regular schedule), irregular (unpredictable or non-recurring), unknown (cannot be classified from available data) |
| `results[].ex_dividend_date` | string | Date when the stock begins trading without the dividend value |
| `results[].frequency` | integer | How many times per year this dividend is expected to occur. A value of 0 means the distribution is non-recurring or irregular (e.g., special, supplemental, or a one-off dividend). Other possible values include 1 (annual), 2 (semi-annual), 3 (trimester), 4 (quarterly), 12 (monthly), 24 (bi-monthly), 52 (weekly), 104 (bi-weekly), and 365 (daily) depending on the issuer's declared or inferred payout cadence. |
| `results[].historical_adjustment_factor` | number | Cumulative adjustment factor used to offset dividend effects on historical prices. To adjust a historical price for dividends: for a price on date D, find the first dividend whose `ex_dividend_date` is after date D and multiply the price by that dividend's `historical_adjustment_factor`. |
| `results[].id` | string | Unique identifier for each dividend record |
| `results[].pay_date` | string | Date when the dividend payment is distributed to shareholders |
| `results[].record_date` | string | Date when shareholders must be on record to be eligible for the dividend payment |
| `results[].split_adjusted_cash_amount` | number | Dividend amount adjusted for stock splits that occurred after the dividend was paid, expressed on a current share basis |
| `results[].ticker` | string | Stock symbol for the company issuing the dividend |
| `status` | enum: OK | The status of this request's response. |

## Sample Response

```json
{
  "request_id": 1,
  "results": [
    {
      "cash_amount": 0.26,
      "currency": "USD",
      "declaration_date": "2025-07-31",
      "distribution_type": "recurring",
      "ex_dividend_date": "2025-08-11",
      "frequency": 4,
      "historical_adjustment_factor": 0.997899,
      "id": "Ed2c9da60abda1e3f0e99a43f6465863c137b671e1f5cd3f833d1fcb4f4eb27fe",
      "pay_date": "2025-08-14",
      "record_date": "2025-08-11",
      "split_adjusted_cash_amount": 0.26,
      "ticker": "AAPL"
    }
  ],
  "status": "OK"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

dividends = []
for d in client.list_stocks_dividends(
    limit="100",
    sort="ticker.asc",
    ):
    dividends.append(d)

print(dividends)