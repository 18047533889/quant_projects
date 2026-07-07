> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### 8-K Text

**Endpoint:** `GET /stocks/filings/8-K/vX/text`

**Description:**

Parsed text content from SEC 8-K filings in plain AI-ready format. 8-K filings are "current reports" that public companies must file with the SEC to announce major events that shareholders should know about, such as acquisitions, leadership changes, or material agreements. The text returned is limited to that in the "Items" sections of the 8-K filings, excluding frontmatter and other boilerplate.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `cik` | string | No | SEC Central Index Key (10 digits, zero-padded). |
| `cik.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `cik.gt` | string | No | Filter greater than the value. |
| `cik.gte` | string | No | Filter greater than or equal to the value. |
| `cik.lt` | string | No | Filter less than the value. |
| `cik.lte` | string | No | Filter less than or equal to the value. |
| `ticker` | string | No | Stock ticker symbol for the company. |
| `ticker.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `ticker.gt` | string | No | Filter greater than the value. |
| `ticker.gte` | string | No | Filter greater than or equal to the value. |
| `ticker.lt` | string | No | Filter less than the value. |
| `ticker.lte` | string | No | Filter less than or equal to the value. |
| `form_type` | string | No | SEC form type (e.g., '8-K', '8-K/A' for amendments). |
| `form_type.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `form_type.gt` | string | No | Filter greater than the value. |
| `form_type.gte` | string | No | Filter greater than or equal to the value. |
| `form_type.lt` | string | No | Filter less than the value. |
| `form_type.lte` | string | No | Filter less than or equal to the value. |
| `filing_date` | string | No | Date when the filing was submitted to the SEC (formatted as YYYY-MM-DD). Value must be formatted 'yyyy-mm-dd'. |
| `filing_date.gt` | string | No | Filter greater than the value. Value must be formatted 'yyyy-mm-dd'. |
| `filing_date.gte` | string | No | Filter greater than or equal to the value. Value must be formatted 'yyyy-mm-dd'. |
| `filing_date.lt` | string | No | Filter less than the value. Value must be formatted 'yyyy-mm-dd'. |
| `filing_date.lte` | string | No | Filter less than or equal to the value. Value must be formatted 'yyyy-mm-dd'. |
| `limit` | integer | No | Limit the maximum number of results returned. Defaults to '100' if not specified. The maximum allowed limit is '999'. |
| `sort` | string | No | A comma separated list of sort columns. For each column, append '.asc' or '.desc' to specify the sort direction. The sort column defaults to 'filing_date' if not specified. The sort order defaults to 'desc' if not specified. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `next_url` | string | If present, this value can be used to fetch the next page. |
| `request_id` | string | A request id assigned by the server. |
| `results` | array[object] | The results for this request. |
| `results[].accession_number` | string | SEC accession number uniquely identifying the filing (e.g., '0000004962-25-000002'). |
| `results[].cik` | string | SEC Central Index Key (10 digits, zero-padded). |
| `results[].filing_date` | string | Date when the filing was submitted to the SEC (formatted as YYYY-MM-DD). |
| `results[].filing_url` | string | SEC URL source for the full filing. |
| `results[].form_type` | string | SEC form type (e.g., '8-K', '8-K/A' for amendments). |
| `results[].items_text` | string | Parsed text content from the 8-K filing, including item numbers and descriptions. |
| `results[].ticker` | string | Stock ticker symbol for the company. |
| `status` | enum: OK | The status of this request's response. |

## Sample Response

```json
{
  "count": 2,
  "next_url": "https://api.massive.com/stocks/filings/8-K/vX/text?cursor=eyJsaW1pd...",
  "request_id": "a3f8b2c1d4e5f6g7",
  "results": [
    {
      "accession_number": "0000004962-25-000002",
      "cik": "0000004962",
      "filing_date": "2025-01-15",
      "filing_url": "https://www.sec.gov/Archives/edgar/data/4962/0000004962-25-000002.txt",
      "form_type": "8-K",
      "items_text": "Item 7.01\tRegulation FD Disclosure\n\nAmerican Express Company is hereby furnishing below delinquency and write-off statistics...",
      "ticker": "AXP"
    },
    {
      "accession_number": "0000320193-25-000010",
      "cik": "0000320193",
      "filing_date": "2025-01-14",
      "filing_url": "https://www.sec.gov/Archives/edgar/data/320193/0000320193-25-000010.txt",
      "form_type": "8-K",
      "items_text": "Item 2.02\tResults of Operations and Financial Condition\n\nOn January 14, 2025, Apple Inc. announced financial results...",
      "ticker": "AAPL"
    }
  ],
  "status": "OK"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

filings = []
for f in client.list_stocks_filings_8k_text(
    limit="100",
    sort="filing_date.desc"
):
    filings.append(f)

print(filings)
