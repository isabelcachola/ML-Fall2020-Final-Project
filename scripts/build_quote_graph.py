'''
Gets all datafiles and updates data/all_tweet_ids.txt
'''
import argparse
import logging
import time
import os
import glob
import pandas as pd
import tqdm
import json
import multiprocessing 
import csv

def convert_data(x):
    return json.loads(x.strip())

def build_quote_graph(fname, num_cores=1):
    fname = glob.glob(os.path.join(os.path.abspath(datadir), "*.jsonl"))[]
    logging.info(f'Found {len(files)} files')

    data = open(fname).readlines()
    with multiprocessing.Pool(num_cores) as mp:
        data = mp.map(convert_data, tqdm.tqdm(data))
    df = pd.DataFrame.from_records(data)

    # Drop non english
    df = df[df['lang'] == 'en']

    # Build graph
    is_quote_tweet =  df['quoted_status'].notna()
    df, quoted = df[~is_quote_tweet],  df[is_quote_tweet]
    df['quote_tweets'] = None

    for _, row in quoted.iterrows():
        if row['quoted_status_id'] in df['id'].values:
            # import ipdb; ipdb.set_trace()
            row_idx = df[df['id'] == row['quoted_status_id']].index[0]
            if df.loc[row_idx]['quote_tweets']:
                df.at[row_idx, 'quote_tweets'].append(row['text'])
            else:
                df.at[row_idx, 'quote_tweets'] = [row['text']]
    # Drop tweets without quote tweets
    df = df[~df['quote_tweets'].isna()]
    return df

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='../data')
    parser.add_argument('--num_cores', type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    df = build_quote_graph(args.datadir, num_cores=args.num_cores)
    df.to_csv('all_data.tsv', sep="\t", quoting=csv.QUOTE_NONNUMERIC, index=False)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')