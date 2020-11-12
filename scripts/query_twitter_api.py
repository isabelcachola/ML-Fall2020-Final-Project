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
import requests
from datetime import datetime
from CREDS import *


def getTweetDetailsWithId(ids):
    headers = {"Authorization": "Bearer {}".format(BEARER_TOKEN)}
    inputs = {'ids': ",".join(ids), 'tweet.fields': 'author_id,public_metrics,entities',
              'expansions': 'entities.mentions.username'}
    response = requests.get("https://api.twitter.com/2/tweets", headers=headers,
                            params=inputs)
    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )

    data_root = response.json()
    data_list = data_root['data']
    return data_list

def stream(api, N, query, datadir, num_requests=1):
    tweets = []
    for tweet in tqdm.tqdm(tweepy.Cursor(api.search, q=query, count=100, lang='en', since='2016-06-20').items()):
        if not tweet.is_quote_status and not tweet.in_reply_to_status_id and not tweet.text.startswith('RT'):
            j = {}
            # Unpack nested dictionaries
            for key, value in tweet._json.items():
                if type(value)==dict:
                    for sub_key, sub_value in value.items():
                        j[f'{key}-{sub_key}'] = sub_value
                else:
                    j[key] = value
            tweets.append(j)
        if len(tweets) >= N:
            break
        time.sleep(2)

    # dd-mm-YY_H:M:S
    dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    df = pd.DataFrame.from_records(tweets)
    root_dicts = getTweetDetailsWithId(list(df['id'].astype(str)))
    df['reply_count'] = [r['public_metrics']['reply_count'] for r in root_dicts]
    df['quote_count'] = [r['public_metrics']['quote_count'] for r in root_dicts]

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