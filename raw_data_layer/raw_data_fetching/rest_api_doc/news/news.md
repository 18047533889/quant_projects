> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Stocks

### News

**Endpoint:** `GET /v2/reference/news`

**Description:**

Retrieve the most recent news articles related to a specified ticker, along with summaries, source details, and sentiment analysis. This endpoint consolidates relevant financial news in one place, extracting associated tickers, assigning sentiment, and providing direct links to the original sources. By incorporating publisher information, article metadata, and sentiment reasoning, users can quickly gauge market sentiment, stay informed on company developments, and integrate news insights into their trading or research workflows.

Use Cases: Market sentiment analysis, investment research, automated monitoring, and portfolio strategy refinement.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `ticker` | string | No | Specify a case-sensitive ticker symbol. For example, AAPL represents Apple Inc. |
| `published_utc` | N/A | No | Return results published on, before, or after this date. |
| `ticker.gte` | string | No | Search by ticker. |
| `ticker.gt` | string | No | Search by ticker. |
| `ticker.lte` | string | No | Search by ticker. |
| `ticker.lt` | string | No | Search by ticker. |
| `published_utc.gte` | N/A | No | Search by published_utc. |
| `published_utc.gt` | N/A | No | Search by published_utc. |
| `published_utc.lte` | N/A | No | Search by published_utc. |
| `published_utc.lt` | N/A | No | Search by published_utc. |
| `order` | string | No | Order results based on the `sort` field. |
| `limit` | integer | No | Limit the number of results returned, default is 10 and max is 1000. |
| `sort` | string | No | Sort field used for ordering. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `count` | integer | The total number of results for this request. |
| `next_url` | string | If present, this value can be used to fetch the next page of data. |
| `request_id` | string | A request id assigned by the server. |
| `results` | array[object] | An array of results containing the requested data. |
| `results[].amp_url` | string | The mobile friendly Accelerated Mobile Page (AMP) URL. |
| `results[].article_url` | string | A link to the news article. |
| `results[].author` | string | The article's author. |
| `results[].description` | string | A description of the article. |
| `results[].id` | string | Unique identifier for the article. |
| `results[].image_url` | string | The article's image URL. |
| `results[].insights` | array[object] | The insights related to the article. |
| `results[].keywords` | array[string] | The keywords associated with the article (which will vary depending on the publishing source). |
| `results[].published_utc` | string | The UTC date and time when the article was published, formatted in RFC3339 standard (e.g. YYYY-MM-DDTHH:MM:SSZ). |
| `results[].publisher` | object | Details the source of the news article, including the publisher's name, logo, and homepage URLs. This information helps users identify and access the original source of news content. |
| `results[].tickers` | array[string] | The ticker symbols associated with the article. |
| `results[].title` | string | The title of the news article. |
| `status` | string | The status of this request's response. |

## Sample Response

```json
{
  "count": 1,
  "next_url": "https://api.massive.com:443/v2/reference/news?cursor=eyJsaW1pdCI6MSwic29ydCI6InB1Ymxpc2hlZF91dGMiLCJvcmRlciI6ImFzY2VuZGluZyIsInRpY2tlciI6e30sInB1Ymxpc2hlZF91dGMiOnsiZ3RlIjoiMjAyMS0wNC0yNiJ9LCJzZWFyY2hfYWZ0ZXIiOlsxNjE5NDA0Mzk3MDAwLG51bGxdfQ",
  "request_id": "831afdb0b8078549fed053476984947a",
  "results": [
    {
      "amp_url": "https://m.uk.investing.com/news/stock-market-news/markets-are-underestimating-fed-cuts-ubs-3559968?ampMode=1",
      "article_url": "https://uk.investing.com/news/stock-market-news/markets-are-underestimating-fed-cuts-ubs-3559968",
      "author": "Sam Boughedda",
      "description": "UBS analysts warn that markets are underestimating the extent of future interest rate cuts by the Federal Reserve, as the weakening economy is likely to justify more cuts than currently anticipated.",
      "id": "8ec638777ca03b553ae516761c2a22ba2fdd2f37befae3ab6fdab74e9e5193eb",
      "image_url": "https://i-invdn-com.investing.com/news/LYNXNPEC4I0AL_L.jpg",
      "insights": [
        {
          "sentiment": "positive",
          "sentiment_reasoning": "UBS analysts are providing a bullish outlook on the extent of future Federal Reserve rate cuts, suggesting that markets are underestimating the number of cuts that will occur.",
          "ticker": "UBS"
        }
      ],
      "keywords": [
        "Federal Reserve",
        "interest rates",
        "economic data"
      ],
      "published_utc": "2024-06-24T18:33:53Z",
      "publisher": {
        "favicon_url": "https://s3.massive.com/public/assets/news/favicons/investing.ico",
        "homepage_url": "https://www.investing.com/",
        "logo_url": "https://s3.massive.com/public/assets/news/logos/investing.png",
        "name": "Investing.com"
      },
      "tickers": [
        "UBS"
      ],
      "title": "Markets are underestimating Fed cuts: UBS By Investing.com - Investing.com UK"
    }
  ],
  "status": "OK"
}
```

from massive import RESTClient
from massive.rest.models import (
    TickerNews,
)

client = RESTClient("0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

news = []
for n in client.list_ticker_news(
	order="asc",
	limit="10",
	sort="published_utc",
	):
    news.append(n)

#print(news)

# print date + title
for index, item in enumerate(news):
    # verify this is an agg
    if isinstance(item, TickerNews):
        print("{:<25}{:<15}".format(item.published_utc, item.title))

        if index == 20:
            break
