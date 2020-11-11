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

def update_unqiue_tweet_ids(datadir):
    files = glob.glob(os.path.join(os.path.abspath(datadir), "*.tsv"))
    logging.info(f'Found {len(files)} files')

    all_ids_path = os.path.join(datadir, 'all_tweet_ids.txt')
    all_ids = set()
    if os.path.exists(all_ids_path):
        lines = open(all_ids_path).readlines()
        for i in lines:
            all_ids.add(i.strip())

    for f in tqdm.tqdm(files):
        df = pd.read_csv(f, sep='\t')
        ids = df['id'].astype(str)
        all_ids = all_ids.union(set(ids))

    with open(all_ids_path, 'w') as fout:
        fout.write('\n'.join(all_ids))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='../data')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    update_unqiue_tweet_ids(args.datadir)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')