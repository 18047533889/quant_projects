> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Risk Factors

**Endpoint:** `GET /stocks/filings/vX/risk-factors`

**Description:**

Standardized, machine-readable risk factor disclosures from SEC filings. Each risk factor is categorized using a consistent taxonomy, enabling direct comparison across time periods or between different companies. For details on the methodology used, see our [research paper](https://arxiv.org/pdf/2601.15247).

Use Cases: Risk trend analysis, cross-company comparisons, compliance tools, classification models, investment research.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `filing_date` | string | No | Date when the filing was submitted to the SEC (formatted as YYYY-MM-DD). |
| `filing_date.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `filing_date.gt` | string | No | Filter greater than the value. |
| `filing_date.gte` | string | No | Filter greater than or equal to the value. |
| `filing_date.lt` | string | No | Filter less than the value. |
| `filing_date.lte` | string | No | Filter less than or equal to the value. |
| `ticker` | string | No | Stock ticker symbol for the company. |
| `ticker.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `ticker.gt` | string | No | Filter greater than the value. |
| `ticker.gte` | string | No | Filter greater than or equal to the value. |
| `ticker.lt` | string | No | Filter less than the value. |
| `ticker.lte` | string | No | Filter less than or equal to the value. |
| `cik` | string | No | SEC Central Index Key (10 digits, zero-padded). |
| `cik.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `cik.gt` | string | No | Filter greater than the value. |
| `cik.gte` | string | No | Filter greater than or equal to the value. |
| `cik.lt` | string | No | Filter less than the value. |
| `cik.lte` | string | No | Filter less than or equal to the value. |
| `limit` | integer | No | Limit the maximum number of results returned. Defaults to '100' if not specified. The maximum allowed limit is '49999'. |
| `sort` | string | No | A comma separated list of sort columns. For each column, append '.asc' or '.desc' to specify the sort direction. The sort column defaults to 'filing_date' if not specified. The sort order defaults to 'desc' if not specified. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `next_url` | string | If present, this value can be used to fetch the next page. |
| `request_id` | string | A request id assigned by the server. |
| `results` | array[object] | The results for this request. |
| `results[].cik` | string | SEC Central Index Key (10 digits, zero-padded). |
| `results[].filing_date` | string | Date when the filing was submitted to the SEC (formatted as YYYY-MM-DD). |
| `results[].primary_category` | string | Top-level risk category |
| `results[].secondary_category` | string | Mid-level risk category |
| `results[].supporting_text` | string | Snippet of text to support the given label |
| `results[].tertiary_category` | string | Most specific risk classification |
| `results[].ticker` | string | Stock ticker symbol for the company. |
| `status` | enum: OK | The status of this request's response. |

## Sample Response

```json
{
  "request_id": "c7856101f86c20d855b0ea1c5a6d6efa",
  "results": [
    {
      "cik": "0001005101",
      "filing_date": "2025-09-19",
      "primary_category": "financial_and_market",
      "secondary_category": "credit_and_liquidity",
      "supporting_text": "In addition to the net proceeds we received from our recent equity and debt financings, we may need to raise additional equity or debt financing to continue the development and marketing of our Fintech app, to fund ongoing operations, invest in acquisitions, and for working capital purposes. Our inability to raise such additional financing may limit our ability to continue the development of our Fintech app.",
      "tertiary_category": "access_to_capital_and_financing",
      "ticker": "MGLD"
    }
  ],
  "status": "OK"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

risk_factors = []
for rf in client.list_stocks_filings_risk_factors(
    limit="100",
    sort="filing_date.desc",
    ):
    risk_factors.append(rf)

print(risk_factors)