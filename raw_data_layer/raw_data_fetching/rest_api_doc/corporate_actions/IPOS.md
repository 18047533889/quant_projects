> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Initial Public Offerings (IPOs)

**Endpoint:** `GET /vX/reference/ipos`

**Description:**

Retrieve comprehensive information on Initial Public Offerings (IPOs), including upcoming and historical events, starting from the year 2008. This endpoint provides key details such as issuer name, ticker symbol, security type, IPO date, number of shares offered, expected price ranges, final issue prices, and offering sizes. Users can filter results by IPO status (e.g., pending, new, rumors, historical) to target their research and inform investment decisions.

Use Cases: IPO research, market trend analysis, investment screening, historical event comparison.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `ticker` | string | No | Specify a case-sensitive ticker symbol. For example, TSLA represents Tesla Inc. |
| `us_code` | string | No | Specify a us_code. This is a unique nine-character alphanumeric code that identifies a North American financial security for the purposes of facilitating clearing and settlement of trades. |
| `isin` | string | No | Specify an International Securities Identification Number (ISIN). This is a unique twelve-digit code that is assigned to every security issuance in the world. |
| `listing_date` | string | No | Specify a listing date. This is the first trading date for the newly listed entity. |
| `ipo_status` | string | No | Specify an IPO status. |
| `listing_date.gte` | string | No | Range by listing_date. |
| `listing_date.gt` | string | No | Range by listing_date. |
| `listing_date.lte` | string | No | Range by listing_date. |
| `listing_date.lt` | string | No | Range by listing_date. |
| `order` | string | No | Order results based on the `sort` field. |
| `limit` | integer | No | Limit the number of results returned, default is 10 and max is 1000. |
| `sort` | string | No | Sort field used for ordering. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `next_url` | string | If present, this value can be used to fetch the next page of data. |
| `request_id` | string | A request id assigned by the server. |
| `results` | array[object] | An array of results containing the requested data. |
| `results[].announced_date` | string | The date when the IPO event was announced. |
| `results[].currency_code` | string | Underlying currency of the security. |
| `results[].final_issue_price` | number | The price set by the company and its underwriters before the IPO goes live. |
| `results[].highest_offer_price` | number | The highest price within the IPO price range that the company might use to price the shares. |
| `results[].ipo_status` | enum: direct_listing_process, history, new, pending, postponed, rumor, withdrawn | The status of the IPO event. IPO events start out as status "rumor" or "pending". On listing day, the status changes to "new". After the listing day, the status changes to "history".  The status "direct_listing_process" corresponds to a type of offering where, instead of going through all the IPO processes, the company decides to list its shares directly on an exchange, without using an investment bank or other intermediaries. This is called a direct listing, direct placement, or direct public offering (DPO). |
| `results[].isin` | string | International Securities Identification Number. This is a unique twelve-digit code that is assigned to every security issuance in the world. |
| `results[].issuer_name` | string | Name of issuer. |
| `results[].last_updated` | string | The date when the IPO event was last modified. |
| `results[].listing_date` | string | First trading date for the newly listed entity. |
| `results[].lot_size` | number | The minimum number of shares that can be bought or sold in a single transaction. |
| `results[].lowest_offer_price` | number | The lowest price within the IPO price range that the company is willing to offer its shares to investors. |
| `results[].max_shares_offered` | number | The upper limit of the shares that the company is offering to investors. |
| `results[].min_shares_offered` | number | The lower limit of shares that the company is willing to sell in the IPO. |
| `results[].primary_exchange` | string | Market Identifier Code (MIC) of the primary exchange where the security is listed. The Market Identifier Code (MIC) (ISO 10383) is a unique identification code used to identify securities trading exchanges, regulated and non-regulated trading markets. |
| `results[].security_description` | string | Description of the security. |
| `results[].security_type` | string | The classification of the stock. For example, "CS" stands for Common Stock. |
| `results[].shares_outstanding` | number | The total number of shares that the company has issued and are held by investors. |
| `results[].ticker` | string | The ticker symbol of the IPO event. |
| `results[].total_offer_size` | number | The total amount raised by the company for IPO. |
| `results[].us_code` | string | This is a unique nine-character alphanumeric code that identifies a North American financial security for the purposes of facilitating clearing and settlement of trades. |
| `status` | string | The status of this request's response. |

## Sample Response

```json
{
  "next_url": "https://api.massive.com/vX/reference/ipos?cursor=YWN0aXZlPXRydWUmZGF0ZT0yMDIxLTA0LTI1JmxpbWl0PTEmb3JkZXI9YXNjJnBhZ2VfbWFya2VyPUElN0M5YWRjMjY0ZTgyM2E1ZjBiOGUyNDc5YmZiOGE1YmYwNDVkYzU0YjgwMDcyMWE2YmI1ZjBjMjQwMjU4MjFmNGZiJnNvcnQ9dGlja2Vy",
  "request_id": "6a7e466379af0a71039d60cc78e72282",
  "results": [
    {
      "announced_date": "2024-06-01",
      "currency_code": "USD",
      "final_issue_price": 17,
      "highest_offer_price": 17,
      "ipo_status": "history",
      "isin": "US75383L1026",
      "issue_end_date": "2024-06-06",
      "issue_start_date": "2024-06-01",
      "issuer_name": "Rapport Therapeutics Inc.",
      "last_updated": "2024-06-27",
      "listing_date": "2024-06-07",
      "lot_size": 100,
      "lowest_offer_price": 17,
      "max_shares_offered": 8000000,
      "min_shares_offered": 1000000,
      "primary_exchange": "XNAS",
      "security_description": "Ordinary Shares",
      "security_type": "CS",
      "shares_outstanding": 35376457,
      "ticker": "RAPP",
      "total_offer_size": 136000000,
      "us_code": "75383L102"
    }
  ],
  "status": "OK"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

ipos = []
for ipo in client.vx.list_ipos(
	order="desc",
	limit=10,
	sort="listing_date",
	):
    ipos.append(ipo)

print(ipos)
