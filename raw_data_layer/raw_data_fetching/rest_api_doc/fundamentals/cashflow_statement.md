> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Cash Flow Statements

**Endpoint:** `GET /stocks/financials/v1/cash-flow-statements`

**Description:**

Retrieve comprehensive cash flow statement data for public companies, including quarterly, annual, and trailing twelve-month cash flows. This dataset includes detailed operating, investing, and financing cash flows with TTM calculations that sum all cash flow components over four quarters.

Use Cases: Cash flow analysis, liquidity assessment, operational efficiency evaluation, investment activity tracking.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `cik` | string | No | The company's Central Index Key (CIK), a unique identifier assigned by the U.S. Securities and Exchange Commission (SEC). You can look up a company’s CIK using the [SEC CIK Lookup tool](https://www.sec.gov/search-filings/cik-lookup). |
| `cik.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `cik.gt` | string | No | Filter greater than the value. |
| `cik.gte` | string | No | Filter greater than or equal to the value. |
| `cik.lt` | string | No | Filter less than the value. |
| `cik.lte` | string | No | Filter less than or equal to the value. |
| `period_end` | string | No | The last date of the reporting period (formatted as YYYY-MM-DD). Value must be formatted 'yyyy-mm-dd'. |
| `period_end.gt` | string | No | Filter greater than the value. Value must be formatted 'yyyy-mm-dd'. |
| `period_end.gte` | string | No | Filter greater than or equal to the value. Value must be formatted 'yyyy-mm-dd'. |
| `period_end.lt` | string | No | Filter less than the value. Value must be formatted 'yyyy-mm-dd'. |
| `period_end.lte` | string | No | Filter less than or equal to the value. Value must be formatted 'yyyy-mm-dd'. |
| `filing_date` | string | No | The date when the financial statement was filed with the SEC. Value must be formatted 'yyyy-mm-dd'. |
| `filing_date.gt` | string | No | Filter greater than the value. Value must be formatted 'yyyy-mm-dd'. |
| `filing_date.gte` | string | No | Filter greater than or equal to the value. Value must be formatted 'yyyy-mm-dd'. |
| `filing_date.lt` | string | No | Filter less than the value. Value must be formatted 'yyyy-mm-dd'. |
| `filing_date.lte` | string | No | Filter less than or equal to the value. Value must be formatted 'yyyy-mm-dd'. |
| `tickers` | string | No | Filter for arrays that contain the value. |
| `tickers.all_of` | string | No | Filter for arrays that contain all of the values. Multiple values can be specified by using a comma separated list. |
| `tickers.any_of` | string | No | Filter for arrays that contain any of the values. Multiple values can be specified by using a comma separated list. |
| `fiscal_year` | number | No | The fiscal year for the reporting period. Value must be a floating point number. |
| `fiscal_year.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `fiscal_year.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `fiscal_year.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `fiscal_year.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `fiscal_quarter` | number | No | The fiscal quarter number (1, 2, 3, or 4) for the reporting period. Value must be a floating point number. |
| `fiscal_quarter.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `fiscal_quarter.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `fiscal_quarter.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `fiscal_quarter.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `timeframe` | string | No | The reporting period type. Possible values include: quarterly, annual, trailing_twelve_months. |
| `timeframe.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `timeframe.gt` | string | No | Filter greater than the value. |
| `timeframe.gte` | string | No | Filter greater than or equal to the value. |
| `timeframe.lt` | string | No | Filter less than the value. |
| `timeframe.lte` | string | No | Filter less than or equal to the value. |
| `max_ticker` | string | No | Filter equal to the value. |
| `max_ticker.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `max_ticker.gt` | string | No | Filter greater than the value. |
| `max_ticker.gte` | string | No | Filter greater than or equal to the value. |
| `max_ticker.lt` | string | No | Filter less than the value. |
| `max_ticker.lte` | string | No | Filter less than or equal to the value. |
| `min_ticker` | string | No | Filter equal to the value. |
| `min_ticker.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `min_ticker.gt` | string | No | Filter greater than the value. |
| `min_ticker.gte` | string | No | Filter greater than or equal to the value. |
| `min_ticker.lt` | string | No | Filter less than the value. |
| `min_ticker.lte` | string | No | Filter less than or equal to the value. |
| `limit` | integer | No | Limit the maximum number of results returned. Defaults to '100' if not specified. The maximum allowed limit is '50000'. |
| `sort` | string | No | A comma separated list of sort columns. For each column, append '.asc' or '.desc' to specify the sort direction. The sort column defaults to 'period_end' if not specified. The sort order defaults to 'asc' if not specified. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `next_url` | string | If present, this value can be used to fetch the next page. |
| `request_id` | string | A request id assigned by the server. |
| `results` | array[object] | The results for this request. |
| `results[].cash_from_operating_activities_continuing_operations` | number | Cash generated from continuing business operations before discontinued operations. |
| `results[].change_in_cash_and_equivalents` | number | Net change in cash and cash equivalents during the period, representing the sum of operating, investing, and financing cash flows plus currency effects. |
| `results[].change_in_other_operating_assets_and_liabilities_net` | number | Net change in working capital components including accounts receivable, inventory, accounts payable, and other operating items. |
| `results[].cik` | string | The company's Central Index Key (CIK), a unique identifier assigned by the U.S. Securities and Exchange Commission (SEC). You can look up a company’s CIK using the [SEC CIK Lookup tool](https://www.sec.gov/search-filings/cik-lookup). |
| `results[].depreciation_depletion_and_amortization` | number | Non-cash charges for the reduction in value of tangible and intangible assets over time. |
| `results[].dividends` | number | Cash payments to shareholders in the form of dividends, typically reported as negative values. |
| `results[].effect_of_currency_exchange_rate` | number | Impact of foreign exchange rate changes on cash and cash equivalents denominated in foreign currencies. |
| `results[].filing_date` | string | The date when the financial statement was filed with the SEC. |
| `results[].fiscal_quarter` | number | The fiscal quarter number (1, 2, 3, or 4) for the reporting period. |
| `results[].fiscal_year` | number | The fiscal year for the reporting period. |
| `results[].income_loss_from_discontinued_operations` | number | After-tax income or loss from business operations that have been discontinued. |
| `results[].long_term_debt_issuances_repayments` | number | Net cash flows from issuing or repaying long-term debt obligations. |
| `results[].net_cash_from_financing_activities` | number | Total cash generated or used by financing activities, including debt issuance, debt repayment, dividends, and share transactions. |
| `results[].net_cash_from_financing_activities_continuing_operations` | number | Cash flows from financing activities of continuing operations before discontinued operations. |
| `results[].net_cash_from_financing_activities_discontinued_operations` | number | Cash flows from financing activities of discontinued business segments. |
| `results[].net_cash_from_investing_activities` | number | Total cash generated or used by investing activities, including capital expenditures, acquisitions, and asset sales. |
| `results[].net_cash_from_investing_activities_continuing_operations` | number | Cash flows from investing activities of continuing operations before discontinued operations. |
| `results[].net_cash_from_investing_activities_discontinued_operations` | number | Cash flows from investing activities of discontinued business segments. |
| `results[].net_cash_from_operating_activities` | number | Total cash generated or used by operating activities, representing cash flow from core business operations. |
| `results[].net_cash_from_operating_activities_discontinued_operations` | number | Cash flows from operating activities of discontinued business segments. |
| `results[].net_income` | number | Net income used as the starting point for operating cash flow calculations. |
| `results[].noncontrolling_interests` | number | Cash flows related to minority shareholders in consolidated subsidiaries. |
| `results[].other_cash_adjustments` | number | Other miscellaneous adjustments to cash flows not classified elsewhere. |
| `results[].other_financing_activities` | number | Cash flows from financing activities not classified elsewhere, including share repurchases and other equity transactions. |
| `results[].other_investing_activities` | number | Cash flows from investing activities not classified elsewhere, including acquisitions, divestitures, and investments. |
| `results[].other_operating_activities` | number | Other adjustments to reconcile net income to operating cash flow not classified elsewhere. |
| `results[].period_end` | string | The last date of the reporting period (formatted as YYYY-MM-DD). |
| `results[].purchase_of_property_plant_and_equipment` | number | Cash outflows for capital expenditures on fixed assets, typically reported as negative values. |
| `results[].sale_of_property_plant_and_equipment` | number | Cash inflows from disposing of fixed assets, typically reported as positive values. |
| `results[].short_term_debt_issuances_repayments` | number | Net cash flows from issuing or repaying short-term debt obligations. |
| `results[].tickers` | array[string] | A list of ticker symbols under which the company is listed. Multiple symbols may indicate different share classes for the same company. |
| `results[].timeframe` | string | The reporting period type. Possible values include: quarterly, annual, trailing_twelve_months. |
| `status` | enum: OK | The status of this request's response. |

## Sample Response

```json
{
  "request_id": "d6d389ca03a0450d93c4ecb13ff26dae",
  "results": [
    {
      "cash_from_operating_activities_continuing_operations": 27867000000,
      "change_in_cash_and_equivalents": 8107000000,
      "change_in_other_operating_assets_and_liabilities_net": -2034000000,
      "cik": "0000320193",
      "depreciation_depletion_and_amortization": 2830000000,
      "dividends": -3945000000,
      "filing_date": "2025-08-01",
      "fiscal_quarter": 3,
      "fiscal_year": 2025,
      "long_term_debt_issuances_repayments": -1192000000,
      "net_cash_from_financing_activities": -24833000000,
      "net_cash_from_financing_activities_continuing_operations": -24833000000,
      "net_cash_from_investing_activities": 5073000000,
      "net_cash_from_investing_activities_continuing_operations": 5073000000,
      "net_cash_from_operating_activities": 27867000000,
      "net_income": 23434000000,
      "other_financing_activities": -23599000000,
      "other_investing_activities": 8535000000,
      "other_operating_activities": 3637000000,
      "period_end": "2025-06-28",
      "purchase_of_property_plant_and_equipment": -3462000000,
      "short_term_debt_issuances_repayments": 3903000000,
      "tickers": [
        "AAPL"
      ],
      "timeframe": "quarterly"
    }
  ],
  "status": "OK"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

cash_flow_statements = []
for c in client.list_financials_cash_flow_statements(
    limit="100",
    sort="period_end.asc"
):
    cash_flow_statements.append(c)

print(cash_flow_statements)