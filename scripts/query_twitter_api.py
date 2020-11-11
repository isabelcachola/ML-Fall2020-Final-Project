'''
Script to query Twitter API
'''
import argparse
import logging
import time
import os
import tweepy 
import pandas as pd
import pprint
import tqdm
import json
import csv
from datetime import datetime
from CREDS import *

class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        print(status.text)

def stream(api, N, query, datadir):
    tweets = []
    for tweet in tqdm.tqdm(tweepy.Cursor(api.search, q=query, count=100, lang='en', since='2017-06-20').items()):
        # Skip quote retweets and reply tweets
        if not tweet.is_quote_status and not tweet.in_reply_to_status_id:
            tweets.append(tweet._json)
        if len(tweets) >= N:
            break
        time.sleep(2)

    # dd-mm-YY_H:M:S
    dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    df = pd.DataFrame.from_records(tweets)
    save_path = os.path.join(datadir, f'{query}_{dt_string}.tsv')
    df.to_csv(save_path, sep='\t', quoting=csv.QUOTE_NONNUMERIC, index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='Sample N tweets')
    parser.add_argument('--query', '-q', dest='query', type=str, help='Search query', default="#Election2020")
    parser.add_argument('--datadir', '-d', dest='datadir', type=str, help='Directory to store data', default="../data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    stream(api, args.N, args.query, args.datadir)

    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')