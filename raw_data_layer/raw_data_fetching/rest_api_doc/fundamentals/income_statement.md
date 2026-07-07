> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Income Statements

**Endpoint:** `GET /stocks/financials/v1/income-statements`

**Description:**

Retrieve comprehensive income statement data for public companies, including key metrics such as revenue, expenses, and net income for various reporting periods. This dataset provides detailed financial performance data including revenue breakdowns, operating expenses, and profitability metrics across quarterly, annual, and trailing twelve-month periods.

Use Cases: Profitability analysis, revenue trend analysis, expense management evaluation, earnings assessment.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `cik` | string | No | The company's Central Index Key (CIK), a unique identifier assigned by the U.S. Securities and Exchange Commission (SEC). You can look up a company’s CIK using the [SEC CIK Lookup tool](https://www.sec.gov/search-filings/cik-lookup). |
| `cik.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `cik.gt` | string | No | Filter greater than the value. |
| `cik.gte` | string | No | Filter greater than or equal to the value. |
| `cik.lt` | string | No | Filter less than the value. |
| `cik.lte` | string | No | Filter less than or equal to the value. |
| `tickers` | string | No | Filter for arrays that contain the value. |
| `tickers.all_of` | string | No | Filter for arrays that contain all of the values. Multiple values can be specified by using a comma separated list. |
| `tickers.any_of` | string | No | Filter for arrays that contain any of the values. Multiple values can be specified by using a comma separated list. |
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
| `results[].basic_earnings_per_share` | number | Earnings per share calculated using the weighted average number of basic shares outstanding. For TTM records, recalculated as TTM net income divided by average basic shares outstanding over the four quarters. |
| `results[].basic_shares_outstanding` | number | Weighted average number of common shares outstanding during the period, used in basic EPS calculation. For TTM records, represents the average over the four most recent quarters. |
| `results[].cik` | string | The company's Central Index Key (CIK), a unique identifier assigned by the U.S. Securities and Exchange Commission (SEC). You can look up a company’s CIK using the [SEC CIK Lookup tool](https://www.sec.gov/search-filings/cik-lookup). |
| `results[].consolidated_net_income_loss` | number | Total net income or loss for the consolidated entity including all subsidiaries. |
| `results[].cost_of_revenue` | number | Direct costs attributable to the production of goods or services sold, also known as cost of goods sold (COGS). |
| `results[].depreciation_depletion_amortization` | number | Non-cash expenses representing the allocation of asset costs over their useful lives. |
| `results[].diluted_earnings_per_share` | number | Earnings per share calculated using diluted shares outstanding, including the effect of potentially dilutive securities. For TTM records, recalculated as TTM net income divided by average diluted shares outstanding over the four quarters. |
| `results[].diluted_shares_outstanding` | number | Weighted average number of shares outstanding including the dilutive effect of stock options, warrants, and convertible securities. For TTM records, represents the average over the four most recent quarters. |
| `results[].discontinued_operations` | number | After-tax results from business segments that have been or will be disposed of. |
| `results[].ebitda` | number | Earnings before interest, taxes, depreciation, and amortization, a measure of operating performance. |
| `results[].equity_in_affiliates` | number | The company's share of income or losses from equity method investments in affiliated companies. |
| `results[].extraordinary_items` | number | Unusual and infrequent gains or losses that are both unusual in nature and infrequent in occurrence. |
| `results[].filing_date` | string | The date when the financial statement was filed with the SEC. |
| `results[].fiscal_quarter` | number | The fiscal quarter number (1, 2, 3, or 4) for the reporting period. |
| `results[].fiscal_year` | number | The fiscal year for the reporting period. |
| `results[].gross_profit` | number | Revenue minus cost of revenue, representing profit before operating expenses. |
| `results[].income_before_income_taxes` | number | Pre-tax income calculated as operating income plus total other income/expense. |
| `results[].income_taxes` | number | Income tax expense or benefit for the period. |
| `results[].interest_expense` | number | Cost of borrowed funds, including interest on debt and other financing obligations. |
| `results[].interest_income` | number | Income earned from interest-bearing investments and cash equivalents. |
| `results[].net_income_loss_attributable_common_shareholders` | number | Net income or loss available to common shareholders after preferred dividends and noncontrolling interests. |
| `results[].noncontrolling_interest` | number | The portion of net income attributable to minority shareholders in consolidated subsidiaries. |
| `results[].operating_income` | number | Income from operations calculated as gross profit minus total operating expenses, excluding non-operating items. |
| `results[].other_income_expense` | number | Non-operating income and expenses not related to the company's core business operations. |
| `results[].other_operating_expenses` | number | Operating expenses not classified in the main expense categories. |
| `results[].period_end` | string | The last date of the reporting period (formatted as YYYY-MM-DD). |
| `results[].preferred_stock_dividends_declared` | number | Dividends declared on preferred stock during the period. |
| `results[].research_development` | number | Expenses incurred for research and development activities to create new products or improve existing ones. |
| `results[].revenue` | number | Total revenue or net sales for the period, representing the company's gross income from operations. |
| `results[].selling_general_administrative` | number | Expenses related to selling products and general administrative costs not directly tied to production. |
| `results[].tickers` | array[string] | A list of ticker symbols under which the company is listed. Multiple symbols may indicate different share classes for the same company. |
| `results[].timeframe` | string | The reporting period type. Possible values include: quarterly, annual, trailing_twelve_months. |
| `results[].total_operating_expenses` | number | Sum of all operating expenses including cost of revenue, SG&A, R&D, depreciation, and other operating expenses. |
| `results[].total_other_income_expense` | number | Net total of all non-operating income and expenses including interest income, interest expense, and other items. |
| `status` | enum: OK | The status of this request's response. |

## Sample Response

```json
{
  "request_id": "5b3cc7c674b34fdd89034b74500bfab5",
  "results": [
    {
      "basic_earnings_per_share": 1.57,
      "basic_shares_outstanding": 14902886000,
      "cik": "0000320193",
      "consolidated_net_income_loss": 23434000000,
      "cost_of_revenue": 50318000000,
      "diluted_earnings_per_share": 1.57,
      "diluted_shares_outstanding": 14948179000,
      "ebitda": 31032000000,
      "filing_date": "2025-08-01",
      "fiscal_quarter": 3,
      "fiscal_year": 2025,
      "gross_profit": 43718000000,
      "income_before_income_taxes": 28031000000,
      "income_taxes": 4597000000,
      "net_income_loss_attributable_common_shareholders": 23434000000,
      "operating_income": 28202000000,
      "other_income_expense": -171000000,
      "other_operating_expenses": 0,
      "period_end": "2025-06-28",
      "research_development": 8866000000,
      "revenue": 94036000000,
      "selling_general_administrative": 6650000000,
      "tickers": [
        "AAPL"
      ],
      "timeframe": "quarterly",
      "total_operating_expenses": 15516000000,
      "total_other_income_expense": -171000000
    }
  ],
  "status": "OK"
}
```


from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

income_statements = []
for i in client.list_financials_income_statements(
    limit="100",
    sort="period_end.asc"
):
    income_statements.append(i)

print(income_statements)