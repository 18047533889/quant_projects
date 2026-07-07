> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Market Status

**Endpoint:** `GET /v1/marketstatus/now`

**Description:**

Retrieve the current trading status for various exchanges and overall financial markets. This endpoint provides real-time indicators of whether markets are open, closed, or operating in pre-market/after-hours sessions, along with timing details for the current or upcoming trading periods.

Use Cases: Real-time monitoring, algorithm scheduling, UI updates, and operational planning.

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `afterHours` | boolean | Whether or not the market is in post-market hours. |
| `currencies` | object | Contains the status of various currency markets. |
| `currencies.crypto` | string | The status of the crypto market. |
| `currencies.fx` | string | The status of the forex market. |
| `earlyHours` | boolean | Whether or not the market is in pre-market hours. |
| `exchanges` | object | Contains the status of different US stock exchanges (e.g., Nasdaq, NYSE). |
| `exchanges.nasdaq` | string | The status of the Nasdaq market. |
| `exchanges.nyse` | string | The status of the NYSE market. |
| `exchanges.otc` | string | The status of the OTC market. |
| `indicesGroups` | object | Contains the status of various index groups (e.g., MSCI, FTSE Russell). |
| `indicesGroups.cccy` | string | The status of Cboe Streaming Market Indices Cryptocurrency ("CCCY") indices trading hours. |
| `indicesGroups.cgi` | string | The status of Cboe Global Indices ("CGI") trading hours. |
| `indicesGroups.dow_jones` | string | The status of Dow Jones indices trading hours |
| `indicesGroups.ftse_russell` | string | The status of Financial Times Stock Exchange Group ("FTSE") Russell indices trading hours. |
| `indicesGroups.msci` | string | The status of Morgan Stanley Capital International ("MSCI") indices trading hours. |
| `indicesGroups.mstar` | string | The status of Morningstar ("MSTAR") indices trading hours. |
| `indicesGroups.mstarc` | string | The status of Morningstar Customer ("MSTARC") indices trading hours. |
| `indicesGroups.nasdaq` | string | The status of National Association of Securities Dealers Automated Quotations ("Nasdaq") indices trading hours. |
| `indicesGroups.s_and_p` | string | The status of Standard & Poor's ("S&P") indices trading hours. |
| `indicesGroups.societe_generale` | string | The status of Societe Generale indices trading hours. |
| `market` | string | The status of the market as a whole. |
| `serverTime` | string | The current time of the server, returned as a date-time in RFC3339 format. |

## Sample Response

```json
{
  "afterHours": true,
  "currencies": {
    "crypto": "open",
    "fx": "open"
  },
  "earlyHours": false,
  "exchanges": {
    "nasdaq": "extended-hours",
    "nyse": "extended-hours",
    "otc": "closed"
  },
  "market": "extended-hours",
  "serverTime": "2020-11-10T17:37:37-05:00"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

result = client.get_market_status()

print(result)
