import yfinance as yf
import pandas as pd
import datetime
import os

TICKER = "^NSEI"
CSV_FILENAME = "mydataset.csv"
today = datetime.date.today()
next_day = today + datetime.timedelta(days=1)

data = yf.download(TICKER, start=today, end=next_day, interval="1d")

if data.empty:
    print(f"No data for {today}. Market might be closed.")
else:
    data.reset_index(inplace=True)

    if os.path.exists(CSV_FILENAME):
        existing = pd.read_csv(CSV_FILENAME)
        if today.strftime('%Y-%m-%d') in existing['Date'].values:
            print(f"Data for {today} already exists.")
        else:
            updated = pd.concat([existing, data], ignore_index=True)
            updated.to_csv(CSV_FILENAME, index=False)
            print("Data updated.")
    else:
        data.to_csv(CSV_FILENAME, index=False)
        print("New dataset created.")
