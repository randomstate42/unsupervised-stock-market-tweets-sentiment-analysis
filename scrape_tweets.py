import twint
from datetime import datetime
import pandas as pd

from utils import *

TWINT_OUTPUT_PATH = "output/{}.csv"
DATA_FILE_PATH = "stock_tweets.csv"
SCRAPE_TWEETS_DAYS_AGO = 20


def get_search_terms():
    df = pd.read_csv(INPUT_PATH)
    df["search_term"] = df["keywords"].str.replace("; ", " OR ")
    return df["search_term"].tolist()


def get_datetime_to_scrape():
    return (
        get_timestamp_days_before(
            timestamp=datetime.now(), n_days=SCRAPE_TWEETS_DAYS_AGO
        )
    ).strftime(TIMESTAMP_FORMAT)


def scrape_tweets(search_term, ticker_symbol, datetime_to_scrape):
    config = twint.Config()
    config.Search = search_term
    config.Lang = "en"
    config.Since = datetime_to_scrape
    config.Pandas = True
    config.Store_object = True
    config.Store_csv = True
    # config.Store_json = True
    config.Output = TWINT_OUTPUT_PATH.format(ticker_symbol)
    config.Hide_output = True
    config.Count = True
    config.Min_likes = 10
    config.Popular_tweets = False
    config.Links = "include"  # 'include', 'exclude'
    config.Filter_retweets = False
    # config.Limit = 20
    twint.run.Search(config)


def main():
    ticker_symbols = get_ticker_symbols()
    search_terms = get_search_terms()

    datetime_to_scrape = get_datetime_to_scrape()
    print(
        f" --- Scraping tweets created since {convert_time_to_eastern_time(datetime_to_scrape)} ET ---"
    )

    out_df = pd.DataFrame()

    count = 0
    for ticker_symbol, search_term in zip(ticker_symbols, search_terms):
        print(f"{ticker_symbol} --> {search_term}")
        scrape_tweets(search_term, datetime_to_scrape)
        # result_list = twint.output.tweets_list
        tweet_df = twint.storage.panda.Tweets_df
        result_list = twint.storage.panda.Tweets_df.to_dict(orient="records")
        # result_json = json.dumps(result_list)
        # print(result_list)
        out_df.columns = out_df.columns
        out_df = pd.concat([tweet_df, out_df], ignore_index=True)

        if result_list:
            count += len(twint.storage.panda.Tweets_df)
            twint.storage.panda.Tweets_df = None

    out_df.to_csv("output/stock_tweets.csv")

    print(f">> Total number of tweets scraped: {count}")
    write_scraped_count(SOURCE_NAME, count)


if __name__ == "__main__":
    main()
