> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Ticker Events

**Endpoint:** `GET /vX/reference/tickers/{id}/events`

**Description:**

Retrieve a timeline of key events associated with a given ticker, CUSIP, or Composite FIGI. This endpoint is experimental and highlights ticker changes, such as symbol renaming or rebranding, helping users maintain continuity in their records and analyses.

Use Cases: Historical reference for ticker symbol changes, data continuity, and record-keeping.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `id` | string | Yes | Identifier of an asset, which can be a Ticker, CUSIP, or Composite FIGI. Specify a case-sensitive  ticker symbol (e.g. AAPL for Apple Inc). When provided a ticker, events for the entity currently  represented by that ticker are returned. To find events for entities previously associated with a  ticker, obtain the relevant identifier using the [Ticker Details Endpoint](https://massive.com/docs/rest/stocks/tickers/ticker-overview). |
| `types` | string | No | A comma-separated list of the types of event to include. Currently ticker_change is the only supported event_type. Leave blank to return all supported event_types. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `request_id` | string | A request id assigned by the server. |
| `results` | object | Contains the requested event data for the specified ticker. |
| `results.events` | array[object] | An array of event containing the requested data. |
| `results.name` | string | The name of the asset. |
| `status` | string | The status of this request's response. |

## Sample Response

```json
{
  "request_id": "31d59dda-80e5-4721-8496-d0d32a654afe",
  "results": {
    "events": [
      {
        "date": "2022-06-09",
        "ticker_change": {
          "ticker": "META"
        },
        "type": "ticker_change"
      },
      {
        "date": "2012-05-18",
        "ticker_change": {
          "ticker": "FB"
        },
        "type": "ticker_change"
      }
    ],
    "name": "Meta Platforms, Inc. Class A Common Stock"
  },
  "status": "OK"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

events = client.get_ticker_events(
	"META",
	)

print(events)
