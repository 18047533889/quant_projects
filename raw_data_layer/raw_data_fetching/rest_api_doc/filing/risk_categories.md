> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Risk Categories

**Endpoint:** `GET /stocks/taxonomies/vX/risk-factors`

**Description:**

The full taxonomy used to classify risk factors in the Risk Factors API. This includes the complete set of standardized categories applied across filings.

Use Cases: Building classifiers, mapping models to risk categories, UI filters, category-level analytics.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `taxonomy` | number | No | Version identifier (e.g., '1.0', '1.1') for the taxonomy Value must be a floating point number. |
| `taxonomy.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `taxonomy.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `taxonomy.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `taxonomy.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `primary_category` | string | No | Top-level risk category |
| `primary_category.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `primary_category.gt` | string | No | Filter greater than the value. |
| `primary_category.gte` | string | No | Filter greater than or equal to the value. |
| `primary_category.lt` | string | No | Filter less than the value. |
| `primary_category.lte` | string | No | Filter less than or equal to the value. |
| `secondary_category` | string | No | Mid-level risk category |
| `secondary_category.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `secondary_category.gt` | string | No | Filter greater than the value. |
| `secondary_category.gte` | string | No | Filter greater than or equal to the value. |
| `secondary_category.lt` | string | No | Filter less than the value. |
| `secondary_category.lte` | string | No | Filter less than or equal to the value. |
| `tertiary_category` | string | No | Most specific risk classification |
| `tertiary_category.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `tertiary_category.gt` | string | No | Filter greater than the value. |
| `tertiary_category.gte` | string | No | Filter greater than or equal to the value. |
| `tertiary_category.lt` | string | No | Filter less than the value. |
| `tertiary_category.lte` | string | No | Filter less than or equal to the value. |
| `limit` | integer | No | Limit the maximum number of results returned. Defaults to '200' if not specified. The maximum allowed limit is '999'. |
| `sort` | string | No | A comma separated list of sort columns. For each column, append '.asc' or '.desc' to specify the sort direction. The sort column defaults to 'taxonomy' if not specified. The sort order defaults to 'desc' if not specified. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `next_url` | string | If present, this value can be used to fetch the next page. |
| `request_id` | string | A request id assigned by the server. |
| `results` | array[object] | The results for this request. |
| `results[].description` | string | Detailed explanation of what this risk category encompasses, including specific examples and potential impacts |
| `results[].primary_category` | string | Top-level risk category |
| `results[].secondary_category` | string | Mid-level risk category |
| `results[].taxonomy` | number | Version identifier (e.g., '1.0', '1.1') for the taxonomy |
| `results[].tertiary_category` | string | Most specific risk classification |
| `status` | enum: OK | The status of this request's response. |

## Sample Response

```json
{
  "request_id": "daac836f71724420b66011d55d88b30b",
  "results": [
    {
      "description": "Risk from inadequate performance management systems, unclear accountability structures, or ineffective measurement and incentive systems that could affect employee performance, goal achievement, and organizational effectiveness.",
      "primary_category": "Governance & Stakeholder",
      "secondary_category": "Organizational & Management",
      "taxonomy": "1.0",
      "tertiary_category": "Performance management and accountability"
    },
    {
      "description": "Risk from requirements to monitor, document, and report on compliance with privacy and data protection regulations including risks from compliance program effectiveness, record-keeping requirements, and breach notification obligations.",
      "primary_category": "Regulatory & Compliance",
      "secondary_category": "Data & Privacy",
      "taxonomy": "1.0",
      "tertiary_category": "Compliance monitoring and reporting"
    }
  ],
  "status": "OK"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

taxonomies = []
for t in client.list_stocks_taxonomies_risk_factors(
    limit="200",
    sort="taxonomy.desc",
    ):
    taxonomies.append(t)

print(taxonomies)