> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Balance Sheets

**Endpoint:** `GET /stocks/financials/v1/balance-sheets`

**Description:**

Retrieve comprehensive balance sheet data for public companies, containing quarterly and annual financial positions. This dataset includes detailed asset, liability, and equity positions representing the company's financial position at specific points in time. Balance sheet data represents point-in-time snapshots rather than cumulative flows, showing what the company owns, owes, and shareholders' equity as of each period end date.

Use Cases: Financial analysis, company valuation, asset assessment, debt analysis, equity research.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `cik` | string | No | The company's Central Index Key (CIK), a unique identifier assigned by the U.S. Securities and Exchange Commission (SEC). You can look up a company's CIK using the [SEC CIK Lookup tool](https://www.sec.gov/search-filings/cik-lookup). |
| `cik.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `cik.gt` | string | No | Filter greater than the value. |
| `cik.gte` | string | No | Filter greater than or equal to the value. |
| `cik.lt` | string | No | Filter less than the value. |
| `cik.lte` | string | No | Filter less than or equal to the value. |
| `tickers` | string | No | Filter for arrays that contain the value. |
| `tickers.all_of` | string | No | Filter for arrays that contain all of the values. Multiple values can be specified by using a comma separated list. |
| `tickers.any_of` | string | No | Filter for arrays that contain any of the values. Multiple values can be specified by using a comma separated list. |
| `period_end` | string | No | The last date of the reporting period, representing the specific point in time when the balance sheet snapshot was taken. Value must be formatted 'yyyy-mm-dd'. |
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
| `timeframe` | string | No | The reporting period type. Possible values include: quarterly, annual. |
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
| `results[].accounts_payable` | number | Amounts owed to suppliers and vendors for goods and services purchased on credit. |
| `results[].accrued_and_other_current_liabilities` | number | Current liabilities not classified elsewhere, including accrued expenses, taxes payable, and other obligations due within one year. |
| `results[].accumulated_other_comprehensive_income` | number | Cumulative gains and losses that bypass the income statement, including foreign currency translation adjustments and unrealized gains/losses on securities. |
| `results[].additional_paid_in_capital` | number | Amount received from shareholders in excess of the par or stated value of shares issued. |
| `results[].cash_and_equivalents` | number | Cash on hand and short-term, highly liquid investments that are readily convertible to known amounts of cash. |
| `results[].cik` | string | The company's Central Index Key (CIK), a unique identifier assigned by the U.S. Securities and Exchange Commission (SEC). You can look up a company's CIK using the [SEC CIK Lookup tool](https://www.sec.gov/search-filings/cik-lookup). |
| `results[].commitments_and_contingencies` | number | Disclosed amount related to contractual commitments and potential liabilities that may arise from uncertain future events. |
| `results[].common_stock` | number | Par or stated value of common shares outstanding representing basic ownership in the company. |
| `results[].debt_current` | number | Short-term borrowings and the current portion of long-term debt due within one year. |
| `results[].deferred_revenue_current` | number | Customer payments received in advance for goods or services to be delivered within one year. |
| `results[].filing_date` | string | The date when the financial statement was filed with the SEC. |
| `results[].fiscal_quarter` | number | The fiscal quarter number (1, 2, 3, or 4) for the reporting period. |
| `results[].fiscal_year` | number | The fiscal year for the reporting period. |
| `results[].goodwill` | number | Intangible asset representing the excess of purchase price over fair value of net assets acquired in business combinations. |
| `results[].intangible_assets_net` | number | Intangible assets other than goodwill, including patents, trademarks, and customer relationships, net of accumulated amortization. |
| `results[].inventories` | number | Raw materials, work-in-process, and finished goods held for sale in the ordinary course of business. |
| `results[].long_term_debt_and_capital_lease_obligations` | number | Long-term borrowings and capital lease obligations with maturities greater than one year. |
| `results[].noncontrolling_interest` | number | Equity in consolidated subsidiaries not owned by the parent company, representing minority shareholders' ownership. |
| `results[].other_assets` | number | Non-current assets not classified elsewhere, including long-term investments, deferred tax assets, and other long-term assets. |
| `results[].other_current_assets` | number | Current assets not classified elsewhere, including prepaid expenses, taxes receivable, and other assets expected to be converted to cash within one year. |
| `results[].other_equity` | number | Equity components not classified elsewhere in shareholders' equity. |
| `results[].other_noncurrent_liabilities` | number | Non-current liabilities not classified elsewhere, including deferred tax liabilities, pension obligations, and other long-term liabilities. |
| `results[].period_end` | string | The last date of the reporting period, representing the specific point in time when the balance sheet snapshot was taken. |
| `results[].preferred_stock` | number | Par or stated value of preferred shares outstanding with preferential rights over common stock. |
| `results[].property_plant_equipment_net` | number | Tangible fixed assets used in operations, reported net of accumulated depreciation. |
| `results[].receivables` | number | Amounts owed to the company by customers and other parties, primarily accounts receivable, net of allowances for doubtful accounts. |
| `results[].retained_earnings_deficit` | number | Cumulative net income earned by the company less dividends paid to shareholders since inception. |
| `results[].short_term_investments` | number | Marketable securities and other investments with maturities of one year or less that are not classified as cash equivalents. |
| `results[].tickers` | array[string] | A list of ticker symbols under which the company is listed. Multiple symbols may indicate different share classes for the same company. |
| `results[].timeframe` | string | The reporting period type. Possible values include: quarterly, annual. |
| `results[].total_assets` | number | Sum of all current and non-current assets representing everything the company owns or controls. |
| `results[].total_current_assets` | number | Sum of all current assets expected to be converted to cash, sold, or consumed within one year. |
| `results[].total_current_liabilities` | number | Sum of all liabilities expected to be settled within one year. |
| `results[].total_equity` | number | Sum of all equity components representing shareholders' total ownership interest in the company. |
| `results[].total_equity_attributable_to_parent` | number | Total shareholders' equity attributable to the parent company, excluding noncontrolling interests. |
| `results[].total_liabilities` | number | Sum of all current and non-current liabilities representing everything the company owes. |
| `results[].total_liabilities_and_equity` | number | Sum of total liabilities and total equity, which should equal total assets per the fundamental accounting equation. |
| `results[].treasury_stock` | number | Cost of the company's own shares that have been repurchased and are held in treasury, typically reported as a negative value. |
| `status` | enum: OK | The status of this request's response. |

## Sample Response

```json
{
  "request_id": "d9f86384d43845a4a3d7b79098fb08dd",
  "results": [
    {
      "accounts_payable": 50374000000,
      "accrued_and_other_current_liabilities": 62499000000,
      "accumulated_other_comprehensive_income": -6369000000,
      "cash_and_equivalents": 36269000000,
      "cik": "0000320193",
      "common_stock": 89806000000,
      "debt_current": 19268000000,
      "deferred_revenue_current": 8979000000,
      "filing_date": "2025-08-01",
      "fiscal_quarter": 3,
      "fiscal_year": 2025,
      "inventories": 5925000000,
      "long_term_debt_and_capital_lease_obligations": 82430000000,
      "other_assets": 160496000000,
      "other_current_assets": 14359000000,
      "other_equity": 0,
      "other_noncurrent_liabilities": 42115000000,
      "period_end": "2025-06-28",
      "property_plant_equipment_net": 48508000000,
      "receivables": 46835000000,
      "retained_earnings_deficit": -17607000000,
      "tickers": [
        "AAPL"
      ],
      "timeframe": "quarterly",
      "total_assets": 331495000000,
      "total_current_assets": 103388000000,
      "total_current_liabilities": 141120000000,
      "total_equity": 65830000000,
      "total_equity_attributable_to_parent": 65830000000,
      "total_liabilities": 265665000000,
      "total_liabilities_and_equity": 331495000000
    }
  ],
  "status": "OK"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

balance_sheets = []
for b in client.list_financials_balance_sheets(
    limit="100",
    sort="period_end.asc"
):
    balance_sheets.append(b)

print(balance_sheets)