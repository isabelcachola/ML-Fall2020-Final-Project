'''
Script to query Twitter API

Parses Mark's data directory and update tweet information

May need to run:
    $ export PYTHONIOENCODING=utf8
'''
import argparse
import logging
import time
import os
import pandas as pd
import pprint
import tqdm
import json
import csv
import requests
import glob
import gzip
import multiprocessing
from datetime import datetime
from CREDS import BEARER_TOKEN


def getTweetDetailsWithId(id):
    try:
        headers = {"Authorization": "Bearer {}".format(BEARER_TOKEN)}
        inputs = {'ids': [id], 'tweet.fields': 'author_id,public_metrics,entities',
                'expansions': 'entities.mentions.username'}
        response = requests.get("https://api.twitter.com/2/tweets", headers=headers,
                                params=inputs)
        if response.status_code != 200:
            logging.warning( "Cannot get stream (HTTP {}): {}".format(response.status_code, response.text))
            return []
        elif response.status_code == 429:
            time.wait(60*15) 
            return getTweetDetailsWithId(id)

        data_root = response.json()
        if 'data' not in data_root:
            return []
        data_list = data_root['data']
        return data_list

    except Exception as e:
        logging.warning(e)
        return []

def convert_data(line):
    try:
        _j = json.loads(line.strip())
        # print(_j.keys())
        # Check language
        if 'lang' not in _j:
            tid = ''
        elif _j['lang'] != 'en':
            tid = ''
        else:
            tid = _j['id_str']
    except UnicodeEncodeError:
        tid = ''
    return tid

def read_data(datadir, num_cores=1):
    files = glob.glob(os.path.join(datadir, "*.gz"))
    # files = [os.path.join(datadir, "example.jsonl.gz")]
    for fname in files:
        name = fname.split('/')[-1].split(".")[0]
        data = gzip.open(fname)
        with multiprocessing.Pool(processes=4) as pool:
            tweet_ids =list(tqdm.tqdm(pool.imap(convert_data, data), desc=f"Reading file {name}"))
        logging.info(f'{len(tweet_ids)} english tweets in file {name}')
        yield (name, tweet_ids)

def chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def save_data(data, name, outdir):
    df = pd.DataFrame.from_records(data)
    save_path = os.path.join(outdir, f'{name}.tsv')
    logging.info(f"Saving {save_path}")
    df.to_csv(save_path, 
            sep='\t', 
            quoting=csv.QUOTE_NONNUMERIC, 
            index=False)

def stream(datadir, outdir, cache_rate=1000, num_cores=1):
    data = read_data(datadir, num_cores=num_cores)
    for d in tqdm.tqdm(data, desc="Total files"):
        name, tweet_ids = d
        tweet_ids = list(filter(lambda x: x!='', tweet_ids))
        data = []
        save_idx = 0
        for i in tqdm.tqdm(tweet_ids, desc="API"):
            data += getTweetDetailsWithId(i)
            time.sleep(2)
            if len(data) >= cache_rate:
                save_data(data, f'{name}-{save_idx}', outdir)
                save_idx += 1
                data = []

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cores', type=int, default=1, help='Sample N tweets')
    parser.add_argument('--datadir', '-d', dest='datadir', type=str, help='Directory to store data', default="../data")
    parser.add_argument('--outdir', '-o', dest='outdir', type=str, help='Directory to store data', default="../data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()

    stream(args.datadir, args.outdir, num_cores=args.num_cores)

    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')