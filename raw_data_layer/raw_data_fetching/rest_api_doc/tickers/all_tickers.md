> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### All Tickers

**Endpoint:** `GET /v3/reference/tickers`

**Description:**

Retrieve a comprehensive list of ticker symbols supported by Massive across various asset classes (e.g., stocks, indices, forex, crypto). Each ticker entry provides essential details such as symbol, name, market, currency, and active status.

Use Cases: Asset discovery, data integration, filtering/selection, and application development.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `ticker` | string | No | Specify a ticker symbol. Defaults to empty string which queries all tickers. |
| `type` | string | No | Specify the type of the tickers. Find the types that we support via our [Ticker Types API](https://massive.com/docs/rest/stocks/tickers/ticker-types). Defaults to empty string which queries all types. |
| `market` | string | No | Filter by market type. By default all markets are included. |
| `exchange` | string | No | Specify the asset's primary exchange Market Identifier Code (MIC) according to [ISO 10383](https://www.iso20022.org/market-identifier-codes). Defaults to empty string which queries all exchanges. |
| `cusip` | string | No | Specify the CUSIP code of the asset you want to search for. Find more information about CUSIP codes [at their website](https://www.cusip.com/identifiers.html#/CUSIP). Defaults to empty string which queries all CUSIPs.  Note: Although you can query by CUSIP, due to legal reasons we do not return the CUSIP in the response. |
| `cik` | string | No | Specify the CIK of the asset you want to search for. Find more information about CIK codes [at their website](https://www.sec.gov/edgar/searchedgar/cik.htm). Defaults to empty string which queries all CIKs. |
| `date` | string | No | Specify a point in time to retrieve tickers available on that date. Defaults to the most recent available date. |
| `search` | string | No | Search for terms within the ticker and/or company name. |
| `active` | boolean | No | Specify if the tickers returned should be actively traded on the queried date. Default is true. |
| `ticker.gte` | string | No | Range by ticker. |
| `ticker.gt` | string | No | Range by ticker. |
| `ticker.lte` | string | No | Range by ticker. |
| `ticker.lt` | string | No | Range by ticker. |
| `order` | string | No | Order results based on the `sort` field. |
| `limit` | integer | No | Limit the number of results returned, default is 100 and max is 1000. |
| `sort` | string | No | Sort field used for ordering. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `count` | integer | The total number of results for this request. |
| `next_url` | string | If present, this value can be used to fetch the next page of data. |
| `request_id` | string | A request id assigned by the server. |
| `results` | array[object] | An array of tickers that match your query.  Note: Although you can query by CUSIP, due to legal reasons we do not return the CUSIP in the response. |
| `results[].active` | boolean | Whether or not the asset is actively traded. False means the asset has been delisted. |
| `results[].base_currency_name` | string | The name of the currency that this asset is priced against. |
| `results[].base_currency_symbol` | string | The ISO 4217 code of the currency that this asset is priced against. |
| `results[].cik` | string | The CIK number for this ticker. Find more information [here](https://en.wikipedia.org/wiki/Central_Index_Key). |
| `results[].composite_figi` | string | The composite OpenFIGI number for this ticker. Find more information [here](https://www.openfigi.com/about/figi) |
| `results[].currency_name` | string | The name of the currency that this asset is traded with. |
| `results[].currency_symbol` | string | The ISO 4217 code of the currency that this asset is traded with. |
| `results[].delisted_utc` | string | The last date that the asset was traded. |
| `results[].last_updated_utc` | string | The information is accurate up to this time. |
| `results[].locale` | enum: us, global | The locale of the asset. |
| `results[].market` | enum: stocks, crypto, fx, otc, indices | The market type of the asset. |
| `results[].name` | string | The name of the asset. For stocks/equities this will be the companies registered name. For crypto/fx this will be the name of the currency or coin pair. |
| `results[].primary_exchange` | string | The ISO code of the primary listing exchange for this asset. |
| `results[].share_class_figi` | string | The share Class OpenFIGI number for this ticker. Find more information [here](https://www.openfigi.com/about/figi) |
| `results[].ticker` | string | The exchange symbol that this item is traded under. |
| `results[].type` | string | The type of the asset. Find the types that we support via our [Ticker Types API](https://massive.com/docs/rest/stocks/tickers/ticker-types). |
| `status` | string | The status of this request's response. |

## Sample Response

```json
{
  "count": 1,
  "next_url": "https://api.massive.com/v3/reference/tickers?cursor=YWN0aXZlPXRydWUmZGF0ZT0yMDIxLTA0LTI1JmxpbWl0PTEmb3JkZXI9YXNjJnBhZ2VfbWFya2VyPUElN0M5YWRjMjY0ZTgyM2E1ZjBiOGUyNDc5YmZiOGE1YmYwNDVkYzU0YjgwMDcyMWE2YmI1ZjBjMjQwMjU4MjFmNGZiJnNvcnQ9dGlja2Vy",
  "request_id": "e70013d92930de90e089dc8fa098888e",
  "results": [
    {
      "active": true,
      "cik": "0001090872",
      "composite_figi": "BBG000BWQYZ5",
      "currency_name": "usd",
      "last_updated_utc": "2021-04-25T00:00:00Z",
      "locale": "us",
      "market": "stocks",
      "name": "Agilent Technologies Inc.",
      "primary_exchange": "XNYS",
      "share_class_figi": "BBG001SCTQY4",
      "ticker": "A",
      "type": "CS"
    }
  ],
  "status": "OK"
}
```


from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

tickers = []
for t in client.list_tickers(
	market="stocks",
	active="true",
	order="asc",
	limit="100",
	sort="ticker",
	):
    tickers.append(t)

print(tickers)
