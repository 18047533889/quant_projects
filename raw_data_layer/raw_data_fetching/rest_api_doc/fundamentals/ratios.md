> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### Ratios

**Endpoint:** `GET /stocks/financials/v1/ratios`

**Description:**

Retrieve comprehensive financial ratios data providing key valuation, profitability, liquidity, and leverage metrics for public companies. This dataset combines data from income statements, balance sheets, and cash flow statements with daily stock prices to calculate ratios for the most recent trading day using trailing twelve months (TTM) financials.

Use Cases: Company valuation, comparative analysis, financial health assessment, investment screening.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `ticker` | string | No | Stock ticker symbol for the company. |
| `ticker.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `ticker.gt` | string | No | Filter greater than the value. |
| `ticker.gte` | string | No | Filter greater than or equal to the value. |
| `ticker.lt` | string | No | Filter less than the value. |
| `ticker.lte` | string | No | Filter less than or equal to the value. |
| `cik` | string | No | Central Index Key (CIK) number assigned by the SEC to identify the company. |
| `cik.any_of` | string | No | Filter equal to any of the values. Multiple values can be specified by using a comma separated list. |
| `cik.gt` | string | No | Filter greater than the value. |
| `cik.gte` | string | No | Filter greater than or equal to the value. |
| `cik.lt` | string | No | Filter less than the value. |
| `cik.lte` | string | No | Filter less than or equal to the value. |
| `price` | number | No | Stock price used in ratio calculations, typically the closing price for the given date. Value must be a floating point number. |
| `price.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `price.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `price.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `price.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `average_volume` | number | No | Average trading volume over the last 30 trading days, providing context for liquidity. Value must be a floating point number. |
| `average_volume.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `average_volume.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `average_volume.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `average_volume.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `market_cap` | number | No | Market capitalization, calculated as stock price multiplied by total shares outstanding. Value must be a floating point number. |
| `market_cap.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `market_cap.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `market_cap.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `market_cap.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `earnings_per_share` | number | No | Earnings per share, calculated as net income available to common shareholders divided by weighted shares outstanding. Value must be a floating point number. |
| `earnings_per_share.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `earnings_per_share.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `earnings_per_share.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `earnings_per_share.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `price_to_earnings` | number | No | Price-to-earnings ratio, calculated as stock price divided by earnings per share. Only calculated when earnings per share is positive. Value must be a floating point number. |
| `price_to_earnings.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `price_to_earnings.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `price_to_earnings.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `price_to_earnings.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `price_to_book` | number | No | Price-to-book ratio, calculated as stock price divided by book value per share, comparing market value to book value. Value must be a floating point number. |
| `price_to_book.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `price_to_book.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `price_to_book.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `price_to_book.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `price_to_sales` | number | No | Price-to-sales ratio, calculated as stock price divided by revenue per share, measuring valuation relative to sales. Value must be a floating point number. |
| `price_to_sales.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `price_to_sales.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `price_to_sales.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `price_to_sales.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `price_to_cash_flow` | number | No | Price-to-cash-flow ratio, calculated as stock price divided by operating cash flow per share. Only calculated when operating cash flow per share is positive. Value must be a floating point number. |
| `price_to_cash_flow.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `price_to_cash_flow.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `price_to_cash_flow.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `price_to_cash_flow.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `price_to_free_cash_flow` | number | No | Price-to-free-cash-flow ratio, calculated as stock price divided by free cash flow per share. Only calculated when free cash flow per share is positive. Value must be a floating point number. |
| `price_to_free_cash_flow.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `price_to_free_cash_flow.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `price_to_free_cash_flow.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `price_to_free_cash_flow.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `dividend_yield` | number | No | Dividend yield, calculated as annual dividends per share divided by stock price, measuring the income return on investment. Value must be a floating point number. |
| `dividend_yield.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `dividend_yield.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `dividend_yield.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `dividend_yield.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `return_on_assets` | number | No | Return on assets ratio, calculated as net income divided by total assets, measuring how efficiently a company uses its assets to generate profit. Value must be a floating point number. |
| `return_on_assets.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `return_on_assets.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `return_on_assets.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `return_on_assets.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `return_on_equity` | number | No | Return on equity ratio, calculated as net income divided by total shareholders' equity, measuring profitability relative to shareholders' equity. Value must be a floating point number. |
| `return_on_equity.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `return_on_equity.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `return_on_equity.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `return_on_equity.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `debt_to_equity` | number | No | Debt-to-equity ratio, calculated as total debt (current debt plus long-term debt) divided by total shareholders' equity, measuring financial leverage. Value must be a floating point number. |
| `debt_to_equity.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `debt_to_equity.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `debt_to_equity.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `debt_to_equity.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `current` | number | No | Current ratio, calculated as total current assets divided by total current liabilities, measuring short-term liquidity. Value must be a floating point number. |
| `current.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `current.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `current.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `current.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `quick` | number | No | Quick ratio (acid-test ratio), calculated as (current assets minus inventories) divided by current liabilities, measuring immediate liquidity. Value must be a floating point number. |
| `quick.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `quick.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `quick.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `quick.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `cash` | number | No | Cash ratio, calculated as cash and cash equivalents divided by current liabilities, measuring the most liquid form of liquidity coverage. Value must be a floating point number. |
| `cash.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `cash.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `cash.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `cash.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `ev_to_sales` | number | No | Enterprise value to sales ratio, calculated as enterprise value divided by revenue, measuring company valuation relative to sales. Value must be a floating point number. |
| `ev_to_sales.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `ev_to_sales.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `ev_to_sales.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `ev_to_sales.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `ev_to_ebitda` | number | No | Enterprise value to EBITDA ratio, calculated as enterprise value divided by EBITDA, measuring company valuation relative to earnings before interest, taxes, depreciation, and amortization. Value must be a floating point number. |
| `ev_to_ebitda.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `ev_to_ebitda.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `ev_to_ebitda.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `ev_to_ebitda.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `enterprise_value` | number | No | Enterprise value, calculated as market capitalization plus total debt minus cash and cash equivalents, representing total company value. Value must be a floating point number. |
| `enterprise_value.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `enterprise_value.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `enterprise_value.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `enterprise_value.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `free_cash_flow` | number | No | Free cash flow, calculated as operating cash flow minus capital expenditures (purchase of property, plant, and equipment). Value must be a floating point number. |
| `free_cash_flow.gt` | number | No | Filter greater than the value. Value must be a floating point number. |
| `free_cash_flow.gte` | number | No | Filter greater than or equal to the value. Value must be a floating point number. |
| `free_cash_flow.lt` | number | No | Filter less than the value. Value must be a floating point number. |
| `free_cash_flow.lte` | number | No | Filter less than or equal to the value. Value must be a floating point number. |
| `limit` | integer | No | Limit the maximum number of results returned. Defaults to '100' if not specified. The maximum allowed limit is '50000'. |
| `sort` | string | No | A comma separated list of sort columns. For each column, append '.asc' or '.desc' to specify the sort direction. The sort column defaults to 'ticker' if not specified. The sort order defaults to 'asc' if not specified. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `next_url` | string | If present, this value can be used to fetch the next page. |
| `request_id` | string | A request id assigned by the server. |
| `results` | array[object] | The results for this request. |
| `results[].average_volume` | number | Average trading volume over the last 30 trading days, providing context for liquidity. |
| `results[].cash` | number | Cash ratio, calculated as cash and cash equivalents divided by current liabilities, measuring the most liquid form of liquidity coverage. |
| `results[].cik` | string | Central Index Key (CIK) number assigned by the SEC to identify the company. |
| `results[].current` | number | Current ratio, calculated as total current assets divided by total current liabilities, measuring short-term liquidity. |
| `results[].date` | string | Date for which the ratios are calculated, representing the trading date with available price data. |
| `results[].debt_to_equity` | number | Debt-to-equity ratio, calculated as total debt (current debt plus long-term debt) divided by total shareholders' equity, measuring financial leverage. |
| `results[].dividend_yield` | number | Dividend yield, calculated as annual dividends per share divided by stock price, measuring the income return on investment. |
| `results[].earnings_per_share` | number | Earnings per share, calculated as net income available to common shareholders divided by weighted shares outstanding. |
| `results[].enterprise_value` | number | Enterprise value, calculated as market capitalization plus total debt minus cash and cash equivalents, representing total company value. |
| `results[].ev_to_ebitda` | number | Enterprise value to EBITDA ratio, calculated as enterprise value divided by EBITDA, measuring company valuation relative to earnings before interest, taxes, depreciation, and amortization. |
| `results[].ev_to_sales` | number | Enterprise value to sales ratio, calculated as enterprise value divided by revenue, measuring company valuation relative to sales. |
| `results[].free_cash_flow` | number | Free cash flow, calculated as operating cash flow minus capital expenditures (purchase of property, plant, and equipment). |
| `results[].market_cap` | number | Market capitalization, calculated as stock price multiplied by total shares outstanding. |
| `results[].price` | number | Stock price used in ratio calculations, typically the closing price for the given date. |
| `results[].price_to_book` | number | Price-to-book ratio, calculated as stock price divided by book value per share, comparing market value to book value. |
| `results[].price_to_cash_flow` | number | Price-to-cash-flow ratio, calculated as stock price divided by operating cash flow per share. Only calculated when operating cash flow per share is positive. |
| `results[].price_to_earnings` | number | Price-to-earnings ratio, calculated as stock price divided by earnings per share. Only calculated when earnings per share is positive. |
| `results[].price_to_free_cash_flow` | number | Price-to-free-cash-flow ratio, calculated as stock price divided by free cash flow per share. Only calculated when free cash flow per share is positive. |
| `results[].price_to_sales` | number | Price-to-sales ratio, calculated as stock price divided by revenue per share, measuring valuation relative to sales. |
| `results[].quick` | number | Quick ratio (acid-test ratio), calculated as (current assets minus inventories) divided by current liabilities, measuring immediate liquidity. |
| `results[].return_on_assets` | number | Return on assets ratio, calculated as net income divided by total assets, measuring how efficiently a company uses its assets to generate profit. |
| `results[].return_on_equity` | number | Return on equity ratio, calculated as net income divided by total shareholders' equity, measuring profitability relative to shareholders' equity. |
| `results[].ticker` | string | Stock ticker symbol for the company. |
| `status` | enum: OK | The status of this request's response. |

## Sample Response

```json
{
  "count": 1,
  "request_id": "8f5374516fec4a819070e53609f47fab",
  "results": [
    {
      "average_volume": 47500000,
      "cash": 0.19,
      "cik": "320193",
      "current": 0.68,
      "date": "2024-09-19",
      "debt_to_equity": 1.52,
      "dividend_yield": 0.0044,
      "earnings_per_share": 6.57,
      "enterprise_value": 3555509835190,
      "ev_to_ebitda": 26.98,
      "ev_to_sales": 9.22,
      "free_cash_flow": 104339000000,
      "market_cap": 3479770835190,
      "price": 228.87,
      "price_to_book": 52.16,
      "price_to_cash_flow": 30.78,
      "price_to_earnings": 34.84,
      "price_to_free_cash_flow": 33.35,
      "price_to_sales": 9.02,
      "quick": 0.63,
      "return_on_assets": 0.3075,
      "return_on_equity": 1.5284,
      "ticker": "AAPL"
    }
  ],
  "status": "OK"
}
```

from massive import RESTClient

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

ratios = []
for r in client.list_financials_ratios(
    limit="100",
    sort="ticker.asc"
):
    ratios.append(r)

print(ratios)