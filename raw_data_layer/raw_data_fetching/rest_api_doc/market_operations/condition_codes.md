> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Condition Codes

**Endpoint:** `GET /v3/reference/conditions`

**Description:**

Retrieve a unified and comprehensive list of trade and quote conditions from various upstream market data providers (e.g., CTA, UTP, OPRA, FINRA). Each condition identifies special circumstances associated with market data, such as trades occurring outside regular sessions or at averaged prices, and outlines how these conditions affect metrics like high, low, open, close, and volume. By examining these mapped conditions, users can accurately interpret the context of trades and quotes, apply appropriate filtering, and ensure that aggregated data correctly reflects eligible trading activity.

Use Cases: Data interpretation, unified condition mapping, filtering and analysis, algorithmic adjustments, compliance and reporting.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `asset_class` | string | No | Filter for conditions within a given asset class. |
| `data_type` | string | No | Filter by data type. |
| `id` | integer | No | Filter for conditions with a given ID. |
| `sip` | string | No | Filter by SIP. If the condition contains a mapping for that SIP, the condition will be returned. |
| `order` | string | No | Order results based on the `sort` field. |
| `limit` | integer | No | Limit the number of results returned, default is 10 and max is 1000. |
| `sort` | string | No | Sort field used for ordering. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `count` | integer | The total number of results for this request. |
| `next_url` | string | If present, this value can be used to fetch the next page of data. |
| `request_id` | string | A request ID assigned by the server. |
| `results` | array[object] | An array of conditions that match your query. |
| `results[].abbreviation` | string | A commonly-used abbreviation for this condition. |
| `results[].asset_class` | enum: stocks, options, crypto, fx | An identifier for a group of similar financial instruments. |
| `results[].data_types` | array[string] | Data types that this condition applies to. |
| `results[].description` | string | A short description of the semantics of this condition. |
| `results[].exchange` | integer | If present, mapping this condition from a Massive code to a SIP symbol depends on this attribute. In other words, data with this condition attached comes exclusively from the given exchange. |
| `results[].id` | integer | An identifier used by Massive for this condition. Unique per data type. |
| `results[].legacy` | boolean | If true, this condition is from an old version of the SIPs' specs and no longer is used. Other conditions may or may not reuse the same symbol as this one. |
| `results[].name` | string | The name of this condition. |
| `results[].sip_mapping` | object | A comprehensive mapping that translates condition codes from individual SIPs (CTA, OPRA, UTP) to a unified code used by Massive. This facilitates consistent interpretation and application of market data conditions across different data streams, ensuring that users can accurately apply these conditions to their data analysis and reporting. |
| `results[].type` | enum: sale_condition, quote_condition, sip_generated_flag, financial_status_indicator, short_sale_restriction_indicator, settlement_condition, market_condition, trade_thru_exempt | An identifier for a collection of related conditions. |
| `results[].update_rules` | object | A list of aggregation rules. |
| `status` | string | The status of this request's response. |





from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

conditions = []
for c in client.list_conditions(
	asset_class="stocks",
	order="asc",
	limit="10",
	sort="asset_class",
	):
    conditions.append(c)

print(conditions)
