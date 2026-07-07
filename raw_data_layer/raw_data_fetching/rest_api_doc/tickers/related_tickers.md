> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Related Tickers

**Endpoint:** `GET /v1/related-companies/{ticker}`

**Description:**

Retrieve a list of tickers related to a specified ticker, identified through an analysis of news coverage and returns data. This endpoint helps users discover peers, competitors, or thematically similar companies, aiding in comparative analysis, portfolio diversification, and market research.

Use Cases: Peer identification, comparative analysis, portfolio diversification, market research.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `ticker` | string | Yes | The ticker symbol to search. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `request_id` | string | A request id assigned by the server. |
| `results` | array[object] | An array of results containing the requested data. |
| `results[].ticker` | string | A ticker related to the requested ticker. |
| `status` | string | The status of this request's response. |
| `ticker` | string | The ticker being queried. |

## Sample Response

```json
{
  "request_id": "31d59dda-80e5-4721-8496-d0d32a654afe",
  "results": [
    {
      "ticker": "MSFT"
    },
    {
      "ticker": "GOOGL"
    },
    {
      "ticker": "AMZN"
    },
    {
      "ticker": "FB"
    },
    {
      "ticker": "TSLA"
    },
    {
      "ticker": "NVDA"
    },
    {
      "ticker": "INTC"
    },
    {
      "ticker": "ADBE"
    },
    {
      "ticker": "NFLX"
    },
    {
      "ticker": "PYPL"
    }
  ],
  "status": "OK",
  "stock_symbol": "AAPL"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

related_companies = client.get_related_companies(
	"AAPL"
	)

print(related_companies)
