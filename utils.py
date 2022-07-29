import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
import json

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

INPUT_PATH = "stocks.csv"

SOURCE_NAME = "twitter"

SCRAPED_COUNT_PATH = "/scraped_count.json"


def get_ticker_symbols():
    df = pd.read_csv(INPUT_PATH)
    return df["ticker_symbol"].tolist()


def convert_time_to_eastern_time(timestamp):
    return datetime.strptime(timestamp, TIMESTAMP_FORMAT) - timedelta(hours=12)


def get_timestamp_days_before(timestamp, n_days):
    return timestamp - timedelta(days=n_days)


def write_scraped_count(data_source, count):
    with open(SCRAPED_COUNT_PATH, "r") as read_file:
        try:
            data = json.load(read_file)
        except Exception as e:
            data = {}
    data[data_source] = count
    with open(SCRAPED_COUNT_PATH, "w") as write_file:
        json.dump(data, write_file)
