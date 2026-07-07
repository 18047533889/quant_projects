> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Float

**Endpoint:** `GET /stocks/vX/float`

**Description:**

Retrieve the latest free float for a specified stock ticker. Free float represents the shares outstanding that are considered available for public trading, after accounting for shares held by strategic or long-term holders.

Shares held by founders, officers and directors, employees, controlling shareholders, affiliated companies, private equity and venture capital firms, sovereign wealth funds, government entities, employee plans, trusts, restricted or locked-up holdings, and any shareholder owning 5% or more of total issued shares are not treated as part of the tradable supply. Shares held by pension funds, mutual funds, ETFs, hedge funds without board representation, depositary banks, and other broadly diversified investors are generally treated as part of the public free float.

Free float reflects a stock’s effective tradable supply and is a key input for assessing liquidity, volatility, market impact, and ownership structure. Figures may reflect ownership changes with a reporting lag, since they are based on publicly available disclosures and ownership information rather than real-time trading activity.

Use cases: liquidity analysis, volatility modeling, position sizing, market impact estimation, and ownership structure analysis.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `ticker` | string | No | The primary ticker symbol for the stock. |
| `ticker.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `ticker.gt` | string | No | Filter greater than the value. |
| `ticker.gte` | string | No | Filter greater than or equal to the value. |
| `ticker.lt` | string | No | Filter less than the value. |
| `ticker.lte` | string | No | Filter less than or equal to the value. |
| `free_float_percent` | number | No | Percentage of total shares outstanding that are available for public trading, rounded to two decimal places. Value must be a floating point number. |
| `free_float_percent.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `free_float_percent.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `free_float_percent.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `free_float_percent.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `limit` | integer | No | Limit the maximum number of results returned. Defaults to '100' if not specified. The maximum allowed limit is '5000'. |
| `sort` | string | No | A comma separated list of sort columns. For each column, append '.asc' or '.desc' to specify the sort direction. The sort column defaults to 'ticker' if not specified. The sort order defaults to 'asc' if not specified. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `next_url` | string | If present, this value can be used to fetch the next page. |
| `request_id` | string | A request id assigned by the server. |
| `results` | array[object] | The results for this request. |
| `results[].effective_date` | string | The effective date of the free float measurement. |
| `results[].free_float` | integer | Number of shares freely tradable in the market. Free float shares represent the portion of a company's outstanding shares that is freely tradable in the market, excluding any holdings considered strategic, controlling, or long term. This excludes insiders, directors, founders, 5 percent plus shareholders, cross holdings, government stakes except pensions, restricted or locked up shares, employee plans, and any entities with board influence, leaving only shares that are genuinely available for public trading. |
| `results[].free_float_percent` | number | Percentage of total shares outstanding that are available for public trading, rounded to two decimal places. |
| `results[].ticker` | string | The primary ticker symbol for the stock. |
| `status` | enum: OK | The status of this request's response. |

## Sample Response

```json
{
  "request_id": 1,
  "results": [
    {
      "effective_date": "2025-11-01",
      "free_float": 15000000000,
      "free_float_percent": 98.5,
      "ticker": "AAPL"
    }
  ],
  "status": "OK"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

floats = []
for f in client.list_stocks_floats(
    limit="100",
    sort="ticker.asc",
    ):
    floats.append(f)

print(floats)

free_float_percent