> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Ticker Overview

**Endpoint:** `GET /v3/reference/tickers/{ticker}`

**Description:**

Retrieve comprehensive details for a single ticker supported by Massive. This endpoint offers a deep look into a company’s fundamental attributes, including its primary exchange, standardized identifiers (CIK, composite FIGI, share class FIGI), market capitalization, industry classification, and key dates. Users also gain access to branding assets (e.g., logos, icons), enabling them to enrich applications and analyses with visually consistent, contextually relevant information.

Use Cases: Company research, data integration, application enhancement, due diligence & compliance.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `ticker` | string | Yes | Specify a case-sensitive ticker symbol. For example, AAPL represents Apple Inc. |
| `date` | string | No | Specify a point in time to get information about the ticker available on that date. When retrieving information from SEC filings, we compare this date with the period of report date on the SEC filing.  For example, consider an SEC filing submitted by AAPL on 2019-07-31, with a period of report date ending on 2019-06-29. That means that the filing was submitted on 2019-07-31, but the filing was created based on information from 2019-06-29. If you were to query for AAPL details on 2019-06-29, the ticker details would include information from the SEC filing.  Defaults to the most recent available date. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `count` | integer | The total number of results for this request. |
| `request_id` | string | A request id assigned by the server. |
| `results` | object | Ticker with details. |
| `results.active` | boolean | Whether or not the asset is actively traded. False means the asset has been delisted. |
| `results.address` | object | Company headquarters address details. |
| `results.branding` | object | Provides URLs aiding in visual identification. |
| `results.cik` | string | The CIK number for this ticker. Find more information [here](https://en.wikipedia.org/wiki/Central_Index_Key). |
| `results.composite_figi` | string | The composite OpenFIGI number for this ticker. Find more information [here](https://www.openfigi.com/about/figi) |
| `results.currency_name` | string | The name of the currency that this asset is traded with. |
| `results.delisted_utc` | string | The last date that the asset was traded. |
| `results.description` | string | A description of the company and what they do/offer. |
| `results.homepage_url` | string | The URL of the company's website homepage. |
| `results.list_date` | string | The date that the symbol was first publicly listed in the format YYYY-MM-DD. |
| `results.locale` | enum: us, global | The locale of the asset. |
| `results.market` | enum: stocks, crypto, fx, otc, indices | The market type of the asset. |
| `results.market_cap` | number | The most recent close price of the ticker multiplied by weighted outstanding shares. |
| `results.name` | string | The name of the asset. For stocks/equities this will be the companies registered name. For crypto/fx this will be the name of the currency or coin pair. |
| `results.phone_number` | string | The phone number for the company behind this ticker. |
| `results.primary_exchange` | string | The ISO code of the primary listing exchange for this asset. |
| `results.round_lot` | number | Round lot size of this security. |
| `results.share_class_figi` | string | The share Class OpenFIGI number for this ticker. Find more information [here](https://www.openfigi.com/about/figi) |
| `results.share_class_shares_outstanding` | number | The recorded number of outstanding shares for this particular share class. |
| `results.sic_code` | string | The standard industrial classification code for this ticker.  For a list of SIC Codes, see the SEC's <a rel="nofollow" target="_blank" href="https://www.sec.gov/info/edgar/siccodes.htm">SIC Code List</a>. |
| `results.sic_description` | string | A description of this ticker's SIC code. |
| `results.ticker` | string | The exchange symbol that this item is traded under. |
| `results.ticker_root` | string | The root of a specified ticker. For example, the root of BRK.A is BRK. |
| `results.ticker_suffix` | string | The suffix of a specified ticker. For example, the suffix of BRK.A is A. |
| `results.total_employees` | number | The approximate number of employees for the company. |
| `results.type` | string | The type of the asset. Find the types that we support via our [Ticker Types API](https://massive.com/docs/rest/stocks/tickers/ticker-types). |
| `results.weighted_shares_outstanding` | number | The shares outstanding calculated assuming all shares of other share classes are converted to this share class. |
| `status` | string | The status of this request's response. |

## Sample Response

```json
{
  "request_id": "31d59dda-80e5-4721-8496-d0d32a654afe",
  "results": {
    "active": true,
    "address": {
      "address1": "One Apple Park Way",
      "city": "Cupertino",
      "postal_code": "95014",
      "state": "CA"
    },
    "branding": {
      "icon_url": "https://api.massive.com/v1/reference/company-branding/d3d3LmFwcGxlLmNvbQ/images/2022-01-10_icon.png",
      "logo_url": "https://api.massive.com/v1/reference/company-branding/d3d3LmFwcGxlLmNvbQ/images/2022-01-10_logo.svg"
    },
    "cik": "0000320193",
    "composite_figi": "BBG000B9XRY4",
    "currency_name": "usd",
    "description": "Apple designs a wide variety of consumer electronic devices, including smartphones (iPhone), tablets (iPad), PCs (Mac), smartwatches (Apple Watch), AirPods, and TV boxes (Apple TV), among others. The iPhone makes up the majority of Apple's total revenue. In addition, Apple offers its customers a variety of services such as Apple Music, iCloud, Apple Care, Apple TV+, Apple Arcade, Apple Card, and Apple Pay, among others. Apple's products run internally developed software and semiconductors, and the firm is well known for its integration of hardware, software and services. Apple's products are distributed online as well as through company-owned stores and third-party retailers. The company generates roughly 40% of its revenue from the Americas, with the remainder earned internationally.",
    "homepage_url": "https://www.apple.com",
    "list_date": "1980-12-12",
    "locale": "us",
    "market": "stocks",
    "market_cap": 2771126040150,
    "name": "Apple Inc.",
    "phone_number": "(408) 996-1010",
    "primary_exchange": "XNAS",
    "round_lot": 100,
    "share_class_figi": "BBG001S5N8V8",
    "share_class_shares_outstanding": 16406400000,
    "sic_code": "3571",
    "sic_description": "ELECTRONIC COMPUTERS",
    "ticker": "AAPL",
    "ticker_root": "AAPL",
    "total_employees": 154000,
    "type": "CS",
    "weighted_shares_outstanding": 16334371000
  },
  "status": "OK"
}
```


from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

details = client.get_ticker_details(
	"AAPL",
	)

print(details)
