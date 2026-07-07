> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Exchanges

**Endpoint:** `GET /v3/reference/exchanges`

**Description:**

Retrieve a list of known exchanges, including their identifiers, names, market types, and other relevant attributes. This information helps map exchange codes, understand market coverage, and integrate exchange details into applications.

Use Cases: Data mapping, market coverage analysis, application development (e.g., display exchange options), and ensuring regulatory compliance.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `asset_class` | string | No | Filter by asset class. |
| `locale` | string | No | Filter by locale. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `count` | integer | The total number of results for this request. |
| `request_id` | string | A request ID assigned by the server. |
| `results` | array[object] | An array of results containing the requested data. |
| `results[].acronym` | string | A commonly used abbreviation for this exchange. |
| `results[].asset_class` | enum: stocks, options, crypto, fx, futures | An identifier for a group of similar financial instruments. |
| `results[].id` | integer | A unique identifier used by Massive for this exchange. |
| `results[].locale` | enum: us, global | An identifier for a geographical location. |
| `results[].mic` | string | The Market Identifier Code of this exchange (see ISO 10383). |
| `results[].name` | string | Name of this exchange. |
| `results[].operating_mic` | string | The MIC of the entity that operates this exchange. |
| `results[].participant_id` | string | The ID used by SIP's to represent this exchange. |
| `results[].type` | enum: exchange, TRF, SIP | Represents the type of exchange. |
| `results[].url` | string | A link to this exchange's website, if one exists. |
| `status` | string | The status of this request's response. |


from massive import RESTClient
from massive.rest.models import (
    Exchange,
)

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

exchanges = client.get_exchanges(
	asset_class="stocks",
	locale="us"
	)

print(exchanges)

# loop over exchanges
for exchange in exchanges:
    # verify this is an exchange
    if isinstance(exchange, Exchange):
        # print exchange info
        print(
            "{:<15}{} ({})".format(
                exchange.asset_class, exchange.name, exchange.operating_mic
            )
        )


from massive import RESTClient
from massive.rest.models import (
    Exchange,
)

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

exchanges = client.get_exchanges(
	asset_class="stocks",
	locale="us"
	)

print(exchanges)

# loop over exchanges
for exchange in exchanges:
    # verify this is an exchange
    if isinstance(exchange, Exchange):
        # print exchange info
        print(
            "{:<15}{} ({})".format(
                exchange.asset_class, exchange.name, exchange.operating_mic
            )
        )
